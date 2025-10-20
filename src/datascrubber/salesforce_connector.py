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
                security_token=sf_config['security_token'],
                instance_type=sf_config.get('instance_type', 'login')
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
            
            # Clean up the records by removing the 'attributes' dictionary
            for record in records:
                del record['attributes']
            
            df = pd.DataFrame(records)
            print(f"-> Successfully fetched {len(df):,} records.")
            return df
        except Exception as e:
            print(f"❌ SOQL Query failed for {query_description}: {e}")
            raise