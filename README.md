# Contact and Account List Scrubber

## Overview

This project provides a robust, command-line tool for scrubbing external contact and account lists against an internal SQL database. Its primary purpose is to identify existing records, prevent the creation of duplicates, and enrich incoming data with internal Account and Contact IDs.

The application is configuration-driven, allowing for easy updates to database credentials, file paths, and matching logic without altering the source code.

---

## Key Features

*   **Dual Scrubbing Modes:** Separate, optimized workflows for scrubbing `account` lists and `contact` lists.
*   **Configurable Matching Logic:** Easily enable or disable matching on fields like street address and website via command-line flags.
*   **Weighted Scoring System:** A sophisticated scoring mechanism, configurable via `config.ini`, determines the "best" match based on customizable weights for different data fields.
*   **Auditing Capability:** Generate an optional "base match" file that shows all potential matches for an input record, not just the best one.
*   **Secure by Design:** Keeps sensitive information like database credentials out of the codebase and Git repository.
*   **Modular and Maintainable:** The project is structured into logical modules for data I/O, normalization, and scrubbing logic, making it easy to extend and maintain.

---

## Project Structure

```
list-scrubber-project/
│
├── .gitignore              # Specifies files and folders for Git to ignore
├── config.example.ini      # Template for the configuration file
├── main.py                 # The single entry point for the application
├── README.md               # This documentation file
├── requirements.txt        # Project dependencies
│
└── src/
    └── datascrubber/
        ├── __init__.py
        ├── data_io.py      # Handles all data reading/writing (SQL, Excel)
        ├── normalization.py# Contains all data cleaning and normalization functions
        ├── scrubbing.py    # Core logic for the scrubbing process
        └── sql_queries.py  # Stores SQL query strings
```

---

## Prerequisites

*   Python 3.8+
*   Git

---

## Setup and Installation

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/erik-kreider/list-scrub-automation
cd list-scrubber-project
```

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required Python libraries using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Configure the Application
The application uses a `config.ini` file for all its settings. This file must be created locally and will be ignored by Git to protect sensitive information.

*   **Create the file:** Copy the provided template.
    ```bash
    cp config.example.ini config.ini
    ```
*   **Edit `config.ini`:** Open the new `config.ini` file and fill in your specific details under each section:
    *   `[Database]`: Add your SQL Server credentials (server, database, username, password).
    *   `[Paths]`: Verify that the input and output directory paths are correct for your machine.
    *   `[Scoring_*]`: Adjust the scoring weights if needed.

> **IMPORTANT:** The `config.ini` file is intentionally listed in `.gitignore`. **NEVER** commit this file to the repository.

---

## Usage

All commands should be run from the root of the project directory (`list-scrubber-project/`).

### Account Scrub
This mode scrubs an external list of accounts.

**Basic Command:**
Provide the `account` mode and the base name of the Excel file (without the `.xlsx` extension).
```bash
python main.py account YourAccountFileName
```

**Command with All Options:**
Enable street matching, website matching, and the creation of an audit file.
```bash
python main.py account YourAccountFileName --street-match --website-match --base-match
```
*   **Output:** This will generate `YourAccountFileName_OUTPUT.xlsx` in the configured output directory.

### Contact Scrub
This mode scrubs a list of contacts. It requires an account scrub to have been run first, as it uses the output from that process as its input.

**Command:**
Provide the `contact` mode and the **original base name** of the file used in the account scrub. The script will automatically look for the `_OUTPUT.xlsx` version.
```bash
python main.py contact YourAccountFileName
```
*   **Input:** The script will automatically use `YourAccountFileName_OUTPUT.xlsx`.
*   **Output:** This will generate `YourAccountFileName_C_OUTPUT.xlsx`.

---

## Configuration Details

The behavior of the scrubber can be fine-tuned in `config.ini`:

*   `[Database]`: Holds all connection details for the source database.
*   `[Paths]`: Defines where the script looks for input Excel files and where it saves the output files.
*   `[Scoring_Account]`: Contains the integer weights for each matching field in the account scrub. Higher numbers give a field more importance.
*   `[Scoring_Contact]`: Contains the integer weights for the contact scrub.
