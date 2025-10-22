import pandas as pd
from simple_salesforce import Salesforce

class SalesforceConnector:
    """Handles all connection and data fetching from the Salesforce API."""

    def __init__(self, config):
        """Initializes the Salesforce connection using credentials from config."""
        sf_config = config['Salesforce']
        try:
            print("Connecting to Salesforce...")
            self.sf = Salesforce(
                username=sf_config['username'],
                password=sf_config['password'],
                security_token=sf_config['security_token']
            )
            print("✅ Salesforce connection successful.")
        except Exception as e:
            print(f"❌ Salesforce connection failed: {e}")
            raise

    def query_to_dataframe(self, soql_query: str, query_description: str) -> pd.DataFrame:
        """
        Executes a SOQL query and returns the results as a pandas DataFrame.
        Handles the conversion from the API's ordered dictionary format.
        """
        print(f"Querying Salesforce for {query_description}...")
        try:
            query_result = self.sf.query_all(soql_query)
            records = query_result['records']
            
            # This recursive function unpacks nested records (like Owner.Name)
            def flatten_records(records):
                flat_list = []
                for record in records:
                    flat_record = {}
                    for key, value in record.items():
                        if isinstance(value, dict) and 'attributes' in value:
                            # Unpack nested object like Owner
                            for nested_key, nested_value in value.items():
                                if nested_key != 'attributes':
                                    flat_record[f"{key}.{nested_key}"] = nested_value
                        else:
                            flat_record[key] = value
                    if 'attributes' in flat_record:
                        del flat_record['attributes']
                    flat_list.append(flat_record)
                return flat_list

            flat_records = flatten_records(records)
            df = pd.DataFrame(flat_records)
            
            print(f"-> Successfully fetched {len(df):,} records.")
            return df
        except Exception as e:
            print(f"❌ SOQL Query failed for {query_description}: {e}")
            raise