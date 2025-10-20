import pandas as pd
import numpy as np
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz
from . import data_io, normalization
from .salesforce_connector import SalesforceConnector
from . import soql_queries

class AccountScrubber:
    """
    Handles the account scrubbing workflow, fetching data directly from Salesforce.
    """
    def __init__(self, config, filename):
        self.config = config
        self.paths = config['Paths']
        self.thresholds = config['Fuzzy_Matching_Thresholds']
        self.weights = config['Scoring_Weights']
        self.penalties = dict(config.items('Scoring_Penalties')) if config.has_section('Scoring_Penalties') else {}
        
        self.filename = filename
        self.input_path = os.path.join(self.paths['input_directory'], f"{filename}.xlsx")
        self.output_path = os.path.join(self.paths['output_directory'], f"{filename}_OUTPUT.xlsx")
        self.manual_review_path = os.path.join(self.paths['output_directory'], f"{filename}_MANUAL_REVIEW.xlsx")

        self.sf_connector = SalesforceConnector(config)

    def _score_candidate(self, scrub_row, db_row):
        """Calculates a fuzzy match score between two account records."""
        score, details = 0, []
        
        penalty = float(self.penalties.get('location_mismatch_penalty', 0))
        
        scrub_country = scrub_row.get('country', '')
        db_country = db_row.get('country', '')
        if scrub_country and db_country and scrub_country != db_country:
            score -= penalty; details.append(f"CountryMismatch(-{penalty:.0f})")

        scrub_state = scrub_row.get('state', '')
        db_state = db_row.get('state', '')
        if scrub_state and db_state and scrub_state != db_state:
            score -= penalty; details.append(f"StateMismatch(-{penalty:.0f})")

        name_sim = fuzz.token_set_ratio(scrub_row.get('normalizedcompany', ''), db_row.get('normalizedcompany', ''))
        name_score = float(self.weights['company_name']) * (name_sim / 100.0)
        if name_score > 1: score += name_score; details.append(f"Name({name_score:.0f})")
        
        scrub_web = scrub_row.get('normalizedwebsite', '')
        db_web = db_row.get('normalizedwebsite', '')
        if scrub_web and db_web and scrub_web == db_web:
            website_score = float(self.weights.get('website', 0))
            score += website_score; details.append(f"Website({website_score:.0f})")

        scrub_phone = scrub_row.get('normalizedphone', '')
        db_phone = db_row.get('normalizedphone', '')
        if scrub_phone and db_phone and scrub_phone == db_phone:
            phone_score = float(self.weights.get('phone', 0))
            score += phone_score; details.append(f"Phone({phone_score:.0f})")

        street_sim = fuzz.ratio(scrub_row.get('normalizedstreet', ''), db_row.get('normalizedstreet', ''))
        street_score = float(self.weights['street']) * (street_sim / 100.0)
        if street_score > 1: score += street_score; details.append(f"Street({street_score:.0f})")
        
        city_sim = fuzz.ratio(scrub_row.get('city', ''), db_row.get('city', ''))
        city_score = float(self.weights.get('city', 0)) * (city_sim / 100.0)
        if city_score > 1: score += city_score; details.append(f"City({city_score:.0f})")

        if scrub_row.get('normalizedpostal') and db_row.get('normalizedpostal') == db_row.get('normalizedpostal'):
            postal_score = float(self.weights['postal_code'])
            score += postal_score; details.append(f"Postal({postal_score:.0f})")

        lob_sim = fuzz.token_set_ratio(scrub_row.get('normalized_lob', ''), db_row.get('normalized_lob', ''))
        lob_score = float(self.weights.get('primary_lob', 0)) * (lob_sim / 100.0)
        if lob_score > 1: score += lob_score; details.append(f"LOB({lob_score:.0f})")

        return score, ",".join(details)

    def run(self):
        """Executes the full account scrubbing workflow using live Salesforce data."""
        start_time = time.time()
        print("\n--- Starting API-Based Account Scrubbing Workflow ---")

        # 1. LOAD DATA
        print("\n[Stage 1/4] Loading local scrub file and fetching Salesforce data...")
        scrub_df = data_io.load_scrub_file(self.input_path)
        
        accounts_df = self.sf_connector.query_to_dataframe(soql_queries.ACCOUNTS_QUERY, "all accounts")
        contacts_df = self.sf_connector.query_to_dataframe(soql_queries.CONTACTS_QUERY, "all contacts")
        
        accounts_df.columns = accounts_df.columns.str.lower()
        contacts_df.columns = contacts_df.columns.str.lower()
        
        scrub_df['original_index'] = scrub_df.index
        original_scrub_df = scrub_df.copy()

        # 2. RENAME & NORMALIZE
        print("\n[Stage 2/4] Renaming and Normalizing Data...")
        scrub_map = {
            'company name': 'company', 'street address': 'street', 'city': 'city', 
            'state': 'state', 'postalcode': 'postal', 'country': 'country',
            'phone': 'phone', 'website domain': 'website', 'primary lob': 'lob'
        }
        accounts_map = {
            'id': 'account_id', 'name': 'company', 'billingstreet': 'street', 
            'billingcity': 'city', 'billingstate': 'state', 'billingpostalcode': 'postal', 
            'billingcountry': 'country', 'phone': 'phone', 'website': 'website',
            'primary_line_of_business__c': 'lob', 'owner.name': 'owner_name', 
            'ownerid': 'owner_id', 'account_status__c': 'account_status', 
            'total_open_opps__c': 'total_open_opps'
        }
        
        scrub_df = scrub_df.rename(columns=scrub_map)
        accounts_df = accounts_df.rename(columns=accounts_map)
        
        for df in [scrub_df, accounts_df]:
            df = normalization.normalize_company(df, 'company')
            df = normalization.normalize_website(df, 'website')
            df = normalization.normalize_phone(df, 'phone')
            df = normalization.normalize_street(df, 'street')
            df = normalization.normalize_postal(df, 'postal')
            df = normalization.normalize_text_field(df, 'city', 'city')
            df = normalization.normalize_text_field(df, 'state', 'state')
            df = normalization.normalize_text_field(df, 'country', 'country')
            df = normalization.normalize_text_field(df, 'lob', 'normalized_lob')
        
        accounts_df['search_string'] = (
            accounts_df['normalizedcompany'].fillna('') + ' ' +
            accounts_df['normalizedwebsite'].fillna('') + ' ' +
            accounts_df['normalizedpostal'].fillna('')
        )

        # 3. PERFORM MATCHING
        print("\n[Stage 3/4] Performing email and fuzzy matching...")
        email_matches_final = pd.DataFrame()
        if 'email' in scrub_df.columns and 'email' in contacts_df.columns:
            scrub_df['email'] = scrub_df['email'].astype(str)
            contacts_df['email'] = contacts_df['email'].astype(str)
            
            email_matched_ids = pd.merge(
                scrub_df[['original_index', 'email']],
                contacts_df[['email', 'accountid']].dropna(subset=['email']).drop_duplicates(subset=['email']),
                on='email', how='inner'
            )
            
            if not email_matched_ids.empty:
                email_matches_details = pd.merge(
                    email_matched_ids, accounts_df,
                    left_on='accountid', right_on='account_id', how='left'
                )
                email_matches_details['match_score'] = 100
                email_matches_details['match_type'] = "Email Match"
                
                email_matches_final = email_matches_details.rename(columns={'account_id': 'matched_accountid', 'company': 'matched_company_name'})
                for col in ['owner_name', 'owner_id', 'account_status', 'total_open_opps', 'lob']:
                    if col not in email_matches_final.columns: email_matches_final[col] = ''

        print(f"-> Found {len(email_matches_final)} records with a direct email match.")
        
        ids_to_skip = email_matches_final['original_index'] if not email_matches_final.empty else []
        fuzzy_search_df = scrub_df[~scrub_df['original_index'].isin(ids_to_skip)]
        
        all_fuzzy_matches = []
        if not fuzzy_search_df.empty:
            vectorizer = TfidfVectorizer(min_df=1, analyzer='char_wb', ngram_range=(3, 5))
            tfidf_matrix = vectorizer.fit_transform(accounts_df['search_string'])

            for _, row in fuzzy_search_df.iterrows():
                row_search_string = (
                    row.get('normalizedcompany', '') + ' ' +
                    row.get('normalizedwebsite', '') + ' ' +
                    row.get('normalizedpostal', '')
                ).strip()
                if not row_search_string: continue
                
                vector = vectorizer.transform([row_search_string])
                similarities = cosine_similarity(vector, tfidf_matrix).flatten()
                top_indices = np.argsort(similarities)[-10:][::-1]
                best_match = None
                highest_score = -1

                for idx in top_indices:
                    candidate = accounts_df.iloc[idx]
                    score, details = self._score_candidate(row, candidate)
                    if score > highest_score:
                        highest_score = score
                        best_match = {
                            'original_index': row['original_index'],
                            'matched_accountid': candidate['account_id'],
                            'match_score': score,
                            'match_type': details,
                            'matched_company_name': candidate.get('company'),
                            'lob': candidate.get('lob'),
                            'owner_name': candidate.get('owner_name'),
                            'owner_id': candidate.get('owner_id'),
                            'account_status': candidate.get('account_status'),
                            'total_open_opps': candidate.get('total_open_opps')
                        }

                if highest_score >= float(self.thresholds['minimum_final_score']):
                    all_fuzzy_matches.append(best_match)
        
        fuzzy_matches_df = pd.DataFrame(all_fuzzy_matches)
        print(f"-> Found {len(fuzzy_matches_df)} records with a confident fuzzy match.")

        # 4. FINALIZE AND SAVE 
        print("\n[Stage 4/4] Finalizing and saving results...")
        final_match_columns = [
            'original_index', 'matched_accountid', 'match_score', 'match_type',
            'matched_company_name', 'lob', 'owner_name', 'owner_id', 
            'account_status', 'total_open_opps'
        ]
        
        email_cols = [col for col in final_match_columns if col in email_matches_final.columns]
        fuzzy_cols = [col for col in final_match_columns if col in fuzzy_matches_df.columns]
        
        final_matches = pd.concat([
            email_matches_final[email_cols] if not email_matches_final.empty else pd.DataFrame(columns=final_match_columns), 
            fuzzy_matches_df[fuzzy_cols] if not fuzzy_matches_df.empty else pd.DataFrame(columns=final_match_columns)
        ], ignore_index=True)
        
        output_df = pd.merge(original_scrub_df, final_matches, on='original_index', how='left')
        
        final_rename_map = {
            'matched_company_name': 'Matched Company Name', 'lob': 'Matched Primary LOB',
            'owner_name': 'Matched Owner Name', 'owner_id': 'Matched Owner ID',
            'account_status': 'Matched Account Status', 'total_open_opps': 'Matched Total Open Opps'
        }
        output_df = output_df.rename(columns=final_rename_map)
        
        output_df.drop(columns=['original_index'], inplace=True, errors='ignore')

        unmatched_ids = scrub_df['original_index']
        if not final_matches.empty:
            unmatched_ids = scrub_df[~scrub_df['original_index'].isin(final_matches['original_index'])]['original_index']
        unmatched_df = original_scrub_df[original_scrub_df['original_index'].isin(unmatched_ids)]
        unmatched_df.drop(columns=['original_index'], inplace=True, errors='ignore')

        data_io.save_to_excel(output_df, self.output_path)
        if not unmatched_df.empty:
            data_io.save_to_excel(unmatched_df, self.manual_review_path)
            print(f"-> {len(unmatched_df)} records require manual review.")

        total_time = time.time() - start_time
        print(f"\n--- Final Workflow Completed in {total_time:.2f} seconds ---")


class ContactScrubber:
    """
    Handles the contact scrubbing workflow, now fetching data directly from Salesforce.
    """

    def __init__(self, config, filename):
        self.config = config
        self.paths = config['Paths']
        self.threshold = float(config.get('Fuzzy_Matching_Thresholds', 'minimum_contact_score', fallback=60))
        self.weights = config['Scoring_Contact'] if 'Scoring_Contact' in config else {}

        self.input_path = os.path.join(self.paths['output_directory'], f"{filename}.xlsx")
        self.output_path = self.input_path.replace('.xlsx', '_C_OUTPUT.xlsx')

        # Initialize the Salesforce connector
        self.sf_connector = SalesforceConnector(config)

    def _score_candidate_contact(self, scrub_row, db_row):
        score, match_details = 0, []
        if scrub_row.get('email') and scrub_row.get('email') == db_row.get('email'):
            email_score = float(self.weights.get('email', 0))
            score += email_score; match_details.append(f"Email({email_score:.0f})")
        sim = fuzz.ratio(scrub_row.get('firstname', ''), db_row.get('firstname', ''))
        name_score = float(self.weights.get('first_name', 0)) * (sim / 100.0)
        if name_score > 0.1: score += name_score; match_details.append(f"First({name_score:.1f})")
        sim = fuzz.ratio(scrub_row.get('lastname', ''), db_row.get('lastname', ''))
        name_score = float(self.weights.get('last_name', 0)) * (sim / 100.0)
        if name_score > 0.1: score += name_score; match_details.append(f"Last({name_score:.1f})")
        sim = fuzz.token_set_ratio(scrub_row.get('title', ''), db_row.get('title', ''))
        title_score = float(self.weights.get('title', 0)) * (sim / 100.0)
        if title_score > 0.1: score += title_score; match_details.append(f"Title({title_score:.1f})")
        return score, ",".join(match_details)

    def run(self):
        """Executes the contact scrubbing workflow using live Salesforce data."""
        total_start_time = time.time()
        print("\n" + "="*50); print("          CONTACT SCRUBBING WORKFLOW START"); print("="*50)

        # 1. LOAD DATA
        print("\n*** STAGE 1: Loading Input Data and Fetching Target Contacts ***")
        scrub_df = data_io.load_scrub_file(self.input_path)
        scrub_df['original_index'] = scrub_df.index
        
        if 'matched_accountid' not in scrub_df.columns:
            raise KeyError("\n\nFATAL ERROR: Input file is missing the 'matched_accountid' column...")

        # Fetch all contacts from Salesforce
        contacts_db = self.sf_connector.query_to_dataframe(soql_queries.CONTACTS_QUERY, "all contacts")
        contacts_db.columns = contacts_db.columns.str.lower()

        matched_account_ids = scrub_df['matched_accountid'].dropna().unique().tolist()
        if not matched_account_ids:
            print("-> No matched Account IDs found. Nothing to scrub.")
            data_io.save_to_excel(scrub_df.drop(columns=['original_index']), self.output_path)
            return

        print(f"-> Found {len(matched_account_ids)} unique Account IDs to target.")

        # 2. PREPARE AND FILTER CONTACT DATABASE
        print("\n*** STAGE 2: Preparing and Filtering Contact Database ***")
        candidate_contacts = contacts_db[contacts_db['accountid'].isin(matched_account_ids)]
        if candidate_contacts.empty:
            print("-> No contacts found in Salesforce for the matched Account IDs.")
            data_io.save_to_excel(scrub_df.drop(columns=['original_index']), self.output_path)
            return
            
        contacts_by_account = candidate_contacts.groupby('accountid').apply(lambda x: x.to_dict('records')).to_dict()
        
        # 3. FIND BEST MATCH FOR EACH CONTACT
        print("\n*** STAGE 3: Finding Best Match for Each Contact ***")
        all_best_matches = []
        records_to_match = scrub_df[scrub_df['matched_accountid'].notna()]
        for _, scrub_row in records_to_match.iterrows():
            account_id = scrub_row['matched_accountid']
            candidates = contacts_by_account.get(account_id, [])
            if not candidates: continue
            best_candidate_details = None
            highest_score = -1
            for candidate_row in candidates:
                score, details = self._score_candidate_contact(scrub_row, candidate_row)
                if score > highest_score:
                    highest_score = score
                    best_candidate_details = {
                        'original_index': scrub_row['original_index'],
                        'Matched_ContactID': candidate_row.get('id'),
                        'Matched_FirstName': candidate_row.get('firstname'),
                        'Matched_LastName': candidate_row.get('lastname'),
                        'Matched_Title': candidate_row.get('title'),
                        'Matched_Email': candidate_row.get('email'),
                        'Matched_ContactPhone': candidate_row.get('phone'),
                        'ContactMatchScore': score,
                        'ContactMatchType': details
                    }
            if highest_score >= self.threshold:
                all_best_matches.append(best_candidate_details)
        
        print(f"-> Found {len(all_best_matches)} confident contact matches.")
        
        # 4. FINALIZE AND SAVE RESULTS
        print("\n*** STAGE 4: Finalizing and Saving Results ***")
        if all_best_matches:
            matches_df = pd.DataFrame(all_best_matches)
            results_df = pd.merge(scrub_df, matches_df, on='original_index', how='left')
        else:
            results_df = scrub_df
        results_df.drop(columns=['original_index'], inplace=True, errors='ignore')
        data_io.save_to_excel(results_df, self.output_path)

        total_time = time.time() - total_start_time
        print(f"\n" + "="*50); print(f"      CONTACT WORKFLOW COMPLETED IN {total_time:.2f} SECONDS"); print("="*50)