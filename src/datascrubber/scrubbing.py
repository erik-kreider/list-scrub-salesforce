import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from . import data_io, normalization, sql_queries
from thefuzz import fuzz

class BaseScrubber:
    """Abstract base class for scrubbing processes."""

    def __init__(self, config, filename):
        self.config = config
        self.filename = filename
        self.input_path = os.path.join(config['Paths']['input_directory'], f"{filename}.xlsx")
        self.output_path = os.path.join(config['Paths']['output_directory'], f"{filename}_OUTPUT.xlsx")

    def run(self):
        """Executes the full scrubbing workflow."""
        raise NotImplementedError("This method should be implemented by a subclass.")

    def _load_scrub_data(self):
        df = data_io.load_scrub_file(self.input_path)
        df['id'] = np.arange(df.shape[0])
        return df

    def _prepare_data(self, df):
        raise NotImplementedError


class AccountScrubber(BaseScrubber):
    """
    Implements a robust "Search and Score" matching engine using TF-IDF
    and Cosine Similarity for candidate selection, followed by fuzzy scoring.
    """

    def __init__(self, config, filename):
        super().__init__(config, filename)
        self.thresholds = config['Fuzzy_Matching_Thresholds']
        self.weights = config['Scoring_Weights']
        if 'Scoring_Penalties' in config:
            self.penalties = config['Scoring_Penalties']
        else:
            self.penalties = {}
        
        # This now correctly reads the lowercase keys from the config file.
        if 'Output_Columns' in config:
            self.output_columns = [key for key, value in config['Output_Columns'].items() if value.lower() == 'true']
        else:
            self.output_columns = []
            
        self.manual_review_path = self.output_path.replace('_OUTPUT.xlsx', '_MANUAL_REVIEW.xlsx')

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares data and creates a focused 'search_string' for vectorization."""
        cols_to_stringify = [
            'DefinitiveID', 'Phone', 'ShippingPostalCode', 'company',
            'ShippingStreet', 'ShippingCity', 'ShippingState', 'LOB', 'Website'
        ]
        for col in cols_to_stringify:
            if col in df.columns:
                df[col] = df[col].astype(str)
        df.fillna('', inplace=True)

        df = normalization.normalize_company(df, 'company')
        df = normalization.normalize_phone(df, 'Phone')
        df = normalization.normalize_website(df, 'Website')
        df = normalization.normalize_street_and_suite(df, 'ShippingStreet')
        df = normalization.normalize_postal(df, 'ShippingPostalCode')

        df['search_string'] = df['NormalizedCompany']
        return df

    def _score_candidate_pair(self, scrub_row, db_row):
        """Calculates a granular, additive fuzzy score to rank candidates."""
        if scrub_row.get('ShippingState') and db_row.get('ShippingState') and scrub_row.get('ShippingState') != db_row.get('ShippingState'):
            return 0, "Knockout(StateMismatch)"

        score = 0
        match_details = []

        if scrub_row.get('NormalizedCompany') and db_row.get('NormalizedCompany'):
            name_sim = fuzz.token_set_ratio(scrub_row['NormalizedCompany'], db_row['NormalizedCompany'])
            name_score = int(self.weights['company_name']) * (name_sim / 100.0)
            score += name_score
            if name_score > 0.1: match_details.append(f"Name({name_score:.1f})")

        if scrub_row.get('NormShipStreet') and db_row.get('NormShipStreet'):
            street_sim = fuzz.ratio(scrub_row['NormShipStreet'], db_row['NormShipStreet'])
            street_score = int(self.weights['street']) * (street_sim / 100.0)
            score += street_score
            if street_score > 0.1: match_details.append(f"Street({street_score:.1f})")

        if scrub_row.get('ShippingCity') and db_row.get('ShippingCity'):
            city_sim = fuzz.ratio(scrub_row['ShippingCity'].lower(), db_row['ShippingCity'].lower())
            city_score = int(self.weights['city']) * (city_sim / 100.0)
            score += city_score
            if city_score > 0.1: match_details.append(f"City({city_score:.1f})")

        if scrub_row.get('NormShipPostal') and db_row.get('NormShipPostal') and scrub_row.get('NormShipPostal') == db_row.get('NormShipPostal'):
            postal_score = int(self.weights['postal_code'])
            score += postal_score
            match_details.append(f"Postal({postal_score})")

        scrub_web = scrub_row.get('NormalizedWebsite', '')
        db_web = db_row.get('NormalizedWebsite', '')
        if scrub_web and db_web and scrub_web != db_web:
            penalty = int(self.penalties.get('conflicting_website_penalty', 0))
            score -= penalty
            if penalty > 0: match_details.append(f"ConflictPenalty(-{penalty})")

        return score, ",".join(match_details)

    def run(self):
        """Executes the final "Vectorize, Search, and Score" workflow."""
        total_start_time = time.time()
        print("\n" + "="*50); print("          ACCOUNT SCRUBBING WORKFLOW START"); print("="*50)

        print("\n*** STAGE 1: Loading and Preparing All Data ***")
        stage_start_time = time.time()
        original_toscrub_df = self._load_scrub_data()
        toscrub_df = self._prepare_data(original_toscrub_df.copy())
        db_df = data_io.load_from_sql(sql_queries.account_sql_query, self.config['Database'])
        db_df = self._prepare_data(db_df.copy())
        print(f"Stage 1 completed in {time.time() - stage_start_time:.2f} seconds.")

        print("\n*** STAGE 2: Building In-Memory Search Matrix ***")
        stage_start_time = time.time()
        vectorizer = TfidfVectorizer(min_df=1, analyzer='char_wb', ngram_range=(3, 5))
        tfidf_matrix_db = vectorizer.fit_transform(db_df['search_string'])
        print(f"Stage 2 completed in {time.time() - stage_start_time:.2f} seconds.")
        
        print("\n*** STAGE 3: Finding Best Matches via Similarity Search ***")
        stage_start_time = time.time()
        all_best_matches = []
        threshold = float(self.thresholds['minimum_final_score'])
        num_candidates_to_score = 10

        for i, scrub_row in enumerate(toscrub_df.to_dict('records'), 1):
            if not scrub_row['search_string'].strip():
                continue

            state_mask = db_df['ShippingState'] == scrub_row['ShippingState']
            state_indices = db_df.index[state_mask]
            
            if len(state_indices) == 0:
                continue

            state_tfidf_matrix = tfidf_matrix_db[state_indices]

            scrub_vector = vectorizer.transform([scrub_row['search_string']])
            cosine_similarities = cosine_similarity(scrub_vector, state_tfidf_matrix).flatten()
            
            num_results = min(num_candidates_to_score, len(cosine_similarities))
            candidate_subset_indices = np.argsort(cosine_similarities)[-num_results:][::-1]
            
            best_candidate = None
            highest_score = -1

            for idx in candidate_subset_indices:
                if cosine_similarities[idx] < 0.1:
                    continue

                original_db_index = state_indices[idx]
                candidate_row = db_df.iloc[original_db_index].to_dict()
                
                score, details = self._score_candidate_pair(scrub_row, candidate_row)
                
                if score > highest_score:
                    highest_score = score
                    best_candidate = candidate_row
                    best_candidate['MatchScore'] = score
                    best_candidate['MatchType'] = details

            if highest_score >= threshold:
                best_candidate['id'] = scrub_row['id']
                all_best_matches.append(best_candidate.copy())

            if i % 1000 == 0:
                print(f"  -> Processed {i:,} of {len(toscrub_df):,}...")

        print(f"-> Found {len(all_best_matches):,} confident matches.")
        print(f"Stage 3 completed in {time.time() - stage_start_time:.2f} seconds.")
        
        print("\n*** STAGE 4: Finalizing and Saving Results ***")
        stage_start_time = time.time()
        
        if all_best_matches:
            matches_df = pd.DataFrame(all_best_matches)
            
            # --- DEFINITIVE FIX: Standardize all columns to lowercase for reliable lookup ---
            matches_df.columns = matches_df.columns.str.lower()
            
            # Define the full rename map using all-lowercase keys
            full_rename_map = {
                'accountid': 'Matched_AccountID', 'company': 'Matched_Company', 'website': 'Matched_Website',
                'phone': 'Matched_Phone', 'lob': 'Matched_LOB', 'account_status__c': 'Matched_Status',
                'team': 'Matched_Team', 'shippingstreet': 'Matched_ShippingStreet',
                'shippingcity': 'Matched_ShippingCity', 'shippingstate': 'Matched_ShippingState',
                'shippingpostalcode': 'Matched_ShippingPostal', 'definitiveid': 'Matched_DefinitiveID'
            }
            matches_df.rename(columns=full_rename_map, inplace=True)

            # Define the final list of columns to show in the output
            final_columns = ['id', 'Matched_AccountID', 'matchscore', 'matchtype']
            for source_col_key in self.output_columns: # e.g., 'shippingstreet'
                if source_col_key in full_rename_map:
                    final_columns.append(full_rename_map[source_col_key])

            # Filter the matches_df to only include this final list of columns
            cols_that_exist = [col for col in final_columns if col in matches_df.columns]
            output_df = matches_df[cols_that_exist]
            
            results_df = pd.merge(original_toscrub_df, output_df, on='id', how='left')
            data_io.save_to_excel(results_df, self.output_path)
            
            matched_ids = matches_df['id'].unique()
            unmatched_df = original_toscrub_df[~original_toscrub_df['id'].isin(matched_ids)]
        else:
            data_io.save_to_excel(original_toscrub_df, self.output_path)
            unmatched_df = original_toscrub_df
            
        if not unmatched_df.empty:
            data_io.save_to_excel(unmatched_df, self.manual_review_path)
        print(f"-> {len(unmatched_df):,} records sent for manual review.")
        print(f"Stage 4 completed in {time.time() - stage_start_time:.2f} seconds.")
        
        print("\n" + "="*50)
        print(f"      WORKFLOW COMPLETED IN {time.time() - total_start_time:.2f} SECONDS")
        print("="*50)

class ContactScrubber(BaseScrubber):
    """
    Implements an intelligent fuzzy matching scrub for contacts, using a highly
    efficient, pre-filtered database query.
    """

    def __init__(self, config, filename):
        super().__init__(config, filename)
        self.threshold = float(config.get('Fuzzy_Matching_Thresholds', 'minimum_contact_score', fallback=60))
        self.weights = config['Scoring_Contact']
        self.output_path = self.output_path.replace('_OUTPUT.xlsx', '_C_OUTPUT.xlsx')

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares contact data using a robust, type-aware method to prevent warnings.
        """
        # --- THE DEFINITIVE FIX for the FutureWarning ---
        # Instead of a global fillna, we fill based on column data type.
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                # If the column is strings/objects, fill missing with an empty string.
                df[col].fillna('', inplace=True)
            elif pd.api.types.is_numeric_dtype(df[col]):
                # If the column is numeric, fill missing with 0.
                df[col].fillna(0, inplace=True)
        
        # Now, proceed with lowercase conversion for the specific columns used in matching.
        cols_to_lower = ['FirstName', 'LastName', 'Email', 'Title', 'firstname', 'lastname', 'email', 'title']
        for col in cols_to_lower:
             if col in df.columns:
                df[col] = df[col].astype(str).str.lower()
                
        return df

    def _score_candidate_pair_contact(self, scrub_row, db_row):
        """Calculates a granular fuzzy score for a potential contact match."""
        score = 0
        match_details = []

        if scrub_row.get('Email', '') and scrub_row.get('Email') == db_row.get('email'):
            email_score = int(self.weights['email'])
            score += email_score
            match_details.append(f"Email({email_score})")

        if scrub_row.get('FirstName', '') and db_row.get('firstname'):
            sim = fuzz.ratio(scrub_row['FirstName'], db_row['firstname'])
            name_score = int(self.weights['first_name']) * (sim / 100.0)
            score += name_score
            if name_score > 0.1: match_details.append(f"First({name_score:.1f})")
            
        if scrub_row.get('LastName', '') and db_row.get('lastname'):
            sim = fuzz.ratio(scrub_row['LastName'], db_row['lastname'])
            name_score = int(self.weights['last_name']) * (sim / 100.0)
            score += name_score
            if name_score > 0.1: match_details.append(f"Last({name_score:.1f})")

        if scrub_row.get('Title', '') and db_row.get('title'):
            sim = fuzz.token_set_ratio(scrub_row['Title'], db_row['title'])
            title_score = int(self.weights['title']) * (sim / 100.0)
            score += title_score
            if title_score > 0.1: match_details.append(f"Title({title_score:.1f})")

        return score, ",".join(match_details)

    def run(self):
        """Executes the new, highly efficient fuzzy matching workflow for contacts."""
        total_start_time = time.time()
        print("\n" + "="*50); print("          CONTACT SCRUBBING WORKFLOW START"); print("="*50)

        print("\n*** STAGE 1: Loading Input Data and Identifying Target Accounts ***")
        stage_start_time = time.time()
        
        original_toscrub_df = self._load_scrub_data()
        toscrub_df = self._prepare_data(original_toscrub_df.copy())
        
        matched_account_ids = toscrub_df[toscrub_df['Matched_AccountID'] != 0]['Matched_AccountID'].dropna().unique().tolist()
        
        if not matched_account_ids:
            print("-> No matched Account IDs found in the input file. Nothing to scrub.")
            data_io.save_to_excel(original_toscrub_df, self.output_path)
            return

        print(f"-> Found {len(matched_account_ids):,} unique Account IDs to query for.")
        print(f"Stage 1 completed in {time.time() - stage_start_time:.2f} seconds.")

        print("\n*** STAGE 2: Executing Targeted SQL Query for Contacts ***")
        stage_start_time = time.time()

        base_query = sql_queries.contact_sql_query
        # Ensure the base query doesn't have a final semicolon
        if base_query.strip().endswith(';'):
            base_query = base_query.strip()[:-1]

        # Use parameters for a safe and correct WHERE IN clause
        placeholders = ','.join('?' for _ in matched_account_ids)
        dynamic_query = f"{base_query} AND a.Id IN ({placeholders})"
        
        # Note: The data_io.py function needs to be able to accept parameters
        # Assuming a modification to data_io.py to handle this, or for simplicity here:
        # A less safe but functional approach if data_io is not modified:
        if len(matched_account_ids) == 1:
            where_in_clause = f"AND a.Id = '{matched_account_ids[0]}'"
        else:
            where_in_clause = f"AND a.Id IN {tuple(matched_account_ids)}"
        dynamic_query = sql_queries.contact_sql_query + " " + where_in_clause

        db_contacts_df = data_io.load_from_sql(dynamic_query, self.config['Database'])
        if db_contacts_df.empty:
            print("-> No contacts found in the database for the matched Account IDs.")
            data_io.save_to_excel(original_toscrub_df, self.output_path)
            return
            
        db_contacts_df.columns = db_contacts_df.columns.str.lower()
        db_contacts_df = self._prepare_data(db_contacts_df)
        print(f"Stage 2 completed in {time.time() - stage_start_time:.2f} seconds.")

        print("\n*** STAGE 3: Grouping Database Contacts ***")
        stage_start_time = time.time()
        
        db_contacts_by_account = db_contacts_df.groupby('accountid').apply(lambda x: x.to_dict('records')).to_dict()
        
        print(f"Stage 3 completed in {time.time() - stage_start_time:.2f} seconds.")

        print("\n*** STAGE 4: Finding Best Match for Each Contact ***")
        stage_start_time = time.time()
        
        records_to_match = toscrub_df[toscrub_df['Matched_AccountID'] != 0].to_dict('records')
        all_best_matches = []

        for _, scrub_row in enumerate(records_to_match, 1):
            matched_account_id = scrub_row.get('Matched_AccountID')
            candidates = db_contacts_by_account.get(matched_account_id, [])
            if not candidates: continue

            best_candidate = None
            highest_score = -1
            for candidate_row in candidates:
                score, details = self._score_candidate_pair_contact(scrub_row, candidate_row)
                if score > highest_score:
                    highest_score = score
                    best_candidate = candidate_row
                    best_candidate['ContactMatchScore'] = score
                    best_candidate['ContactMatchType'] = details

            if highest_score >= self.threshold:
                best_candidate['id'] = scrub_row['id']
                all_best_matches.append(best_candidate.copy())
        
        print(f"-> Found {len(all_best_matches):,} confident contact matches.")
        print(f"Stage 4 completed in {time.time() - stage_start_time:.2f} seconds.")

        print("\n*** STAGE 5: Finalizing and Saving Results ***")
        stage_start_time = time.time()

        if all_best_matches:
            matches_df = pd.DataFrame(all_best_matches)
            rename_map = {
                'contactid': 'Matched_ContactID', 'firstname': 'Matched_FirstName',
                'lastname': 'Matched_LastName', 'title': 'Matched_Title',
                'email': 'Matched_Email', 'phone': 'Matched_ContactPhone'
            }
            matches_df.rename(columns=rename_map, inplace=True)
            
            final_cols = ['id', 'Matched_ContactID', 'Matched_FirstName', 'Matched_LastName', 
                          'Matched_Title', 'Matched_Email', 'Matched_ContactPhone',
                          'ContactMatchScore', 'ContactMatchType']
            cols_that_exist = [col for col in final_cols if col in matches_df.columns]
            output_df = matches_df[cols_that_exist]

            results_df = pd.merge(original_toscrub_df, output_df, on='id', how='left')
        else:
            results_df = original_toscrub_df

        data_io.save_to_excel(results_df, self.output_path)
        print(f"Stage 5 completed in {time.time() - stage_start_time:.2f} seconds.")

        print("\n" + "="*50)
        print(f"      CONTACT WORKFLOW COMPLETED IN {time.time() - total_start_time:.2f} SECONDS")
        print("="*50)