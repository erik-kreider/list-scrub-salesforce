# SOQL Queries for fetching Salesforce data.

# Fetches all relevant fields from Accounts for matching.
ACCOUNTS_QUERY = """
SELECT
    Id,
    Name,
    BillingStreet,
    BillingCity,
    BillingState,
    BillingPostalCode,
    BillingCountry,
    Phone,
    Website,
    Primary_Line_of_Business__c,
    Owner.Name,
    OwnerId,
    Account_Status__c,
    Total_Open_Opps__c
FROM Account
"""

# Fetches all relevant fields from Contacts
CONTACTS_QUERY = """
SELECT
    AccountId,
    Id,
    FirstName,
    LastName,
    Phone,
    Email,
    Title, 
    Job_Function__c
FROM Contact
"""