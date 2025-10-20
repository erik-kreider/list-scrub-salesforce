# Salesforce List Scrubber

## 🚀 Introduction

This project provides a powerful and flexible command-line tool to scrub third-party lead and contact lists against live data exports from Salesforce. Its primary purpose is to identify which records in a new list already exist as Accounts or Contacts in your Salesforce instance, preventing duplicate data entry and enriching incoming leads with existing data.

The tool is designed to be resilient to messy data, using a sophisticated two-stage matching process that combines high-confidence email matching with advanced fuzzy logic scoring.

## ✨ Key Features

-   **Dual Scrubbing Modes**: Perform scrubs for `account` or `contact` records.
-   **Live Salesforce Query-Based**: Operates entirely on live exports from Salesforce, requiring a live database connection.
-   **Advanced Normalization**: Cleans and standardizes various data fields (company names, websites, phone numbers, addresses) before matching to increase accuracy.
-   **Two-Stage Matching Logic**:
    1.  **Email First**: A high-speed, high-confidence pre-match based on contact email addresses.
    2.  **Fuzzy Logic Scoring**: For remaining records, it uses a TF-IDF similarity search to find the best potential candidates, then scores each one based on a configurable weighted system.
-   **Highly Configurable**: Easily control matching thresholds, scoring weights for each data field, and penalties for conflicting data via a simple `config.ini` file.
-   **Flexible Penalty System**: Avoids rigid "knockout" rules by applying configurable penalties for mismatches (e.g., conflicting websites), making it robust against dirty data.
-   **Clear Output**: Generates a primary output file with matched data appended, and a separate file for all unmatched records requiring manual review.
# Salesforce List Scrubber

## What this project does

A command-line tool that scrubs third-party lead and contact lists against Salesforce data. It identifies existing Accounts and Contacts in your Salesforce org and appends matched metadata to your input lists. Matching is performed with an email-first pass followed by a TF-IDF + fuzzy-scoring pipeline.

This repository supports two operating modes:
- account — find matching Salesforce Accounts for companies in a list
- contact — find matching Salesforce Contacts (within previously matched Accounts)

## Notable change: Salesforce API integration

This version fetches Accounts and Contacts directly from Salesforce using the simple-salesforce library. You can run the scrubbers against live Salesforce data instead of static Excel exports. Authentication and connection settings are read from the `Salesforce` section of `config.ini` (see configuration below).

## Quick start

1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv venv; .\venv\Scripts\Activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Configure `config.ini` (example below) with your paths and Salesforce credentials.

4. Put the list you want to scrub into the `lists/` directory (filename without `.xlsx` is passed to the CLI).

5. Run the account scrub:

```powershell
python .\main.py account my_third_party_list
```

Then run the contact scrub against the account output (optional):

```powershell
python .\main.py contact my_third_party_list_OUTPUT
```

## Configuration (`config.ini`)

At minimum update the `Paths` and `Salesforce` sections. The code expects a file named `config.ini` in the repository root.

Example minimal `config.ini`:

```ini
[Paths]
input_directory = ./lists
output_directory = ./lists

[Salesforce]
username = your_username@example.com
password = your_password
security_token = YOUR_SECURITY_TOKEN
instance_type = login   ; optional (login or sandbox)

[Fuzzy_Matching_Thresholds]
minimum_final_score = 60
minimum_contact_score = 45

[Scoring_Weights]
company_name = 50
website = 40
phone = 35
street = 25
postal_code = 15
city = 10
primary_lob = 10

[Scoring_Penalties]
conflicting_website_penalty = 20
location_mismatch_penalty = 20

[Scoring_Contact]
email = 50
first_name = 20
last_name = 30
title = 10
```

Notes:
- `Salesforce` credentials are used to initialize a `simple_salesforce.Salesforce` client. Keep these credentials secure.
- `minimum_final_score` controls how strict account fuzzy-matching is. Tune to your dataset.

## How the API integration works

- On startup, `SalesforceConnector` reads `config.ini` -> `Salesforce` and attempts to connect with simple-salesforce.
- Two SOQL queries (in `src/datascrubber/soql_queries.py`) fetch Account and Contact fields used by the scrubbers.
- The connector method `query_to_dataframe(soql_query, description)` returns a pandas DataFrame with the query results.

If Salesforce authentication fails the tool will raise and print the connector error.

## Inputs and expected columns

- Input lists (in `./lists`) should be Excel (`.xlsx`) files. Column headers are normalized to lowercase by the loader.
- Typical input column names the code expects (normalized):
  - company name, street address, city, state, postalcode, country, phone, website domain, email
- Salesforce Account fields used (from SOQL): id, name, billingstreet, billingcity, billingstate, billingpostalcode, billingcountry, phone, website, primary_line_of_business__c, owner.name, ownerid, account_status__c, total_open_opps__c
- Salesforce Contact fields used (from SOQL): accountid, id, firstname, lastname, email, phone, title

The data loader (`src/datascrubber/data_io.py`) attempts to clean and normalize Salesforce report artifacts if you prefer to use static exports instead of the API.

## Outputs

- Account run: `<filename>_OUTPUT.xlsx` (matched columns appended) and `<filename>_MANUAL_REVIEW.xlsx` for unmatched rows (when present).
- Contact run: `<filename>_C_OUTPUT.xlsx` (contact-level matches appended).

## Developer notes

- Primary modules live in `src/datascrubber/`.
- `AccountScrubber.run()` and `ContactScrubber.run()` encapsulate the main workflows and use `SalesforceConnector` to fetch live data.
- SOQL queries are defined in `src/datascrubber/soql_queries.py` and can be adjusted to include additional fields.

Edge cases to consider when modifying or extending:
- Missing or incorrect Salesforce credentials (authentication failure).
- Very large Salesforce exports: the TF-IDF stage builds an in-memory matrix of Account search strings — watch memory usage for very large orgs.
- Input files missing expected columns (the code normalizes headers but will raise on missing critical columns during contact scrub).

