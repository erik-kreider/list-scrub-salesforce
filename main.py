import argparse
import configparser
import time
from src.datascrubber.scrubbing import AccountScrubber, ContactScrubber

def main():
    """
    Main entry point for the data scrubbing application.
    Parses command-line arguments, loads configuration, and runs the appropriate scrubber.
    """
    # Load configuration from config.ini
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Set up the main command-line parser
    parser = argparse.ArgumentParser(description="Scrub account or contact lists against a database.")
    subparsers = parser.add_subparsers(dest='scrub_type', required=True, help="The type of scrub to perform.")

    # --- Account Scrubber Sub-parser ---
    parser_account = subparsers.add_parser('account', help="Run the account scrub.")
    parser_account.add_argument('filename', type=str, help="The base name of the Excel file to scrub (without .xlsx).")

    # --- Contact Scrubber Sub-parser ---
    parser_contact = subparsers.add_parser('contact', help="Run the contact scrub.")
    parser_contact.add_argument('filename', type=str, help="The base name of the account scrub OUTPUT file to use for contact scrubbing.")

    args = parser.parse_args()
    
    start_time = time.time()
    print(f"Starting {args.scrub_type} scrub for file: {args.filename}...")

    if args.scrub_type == 'account':
        scrubber = AccountScrubber(config, filename=args.filename)
    elif args.scrub_type == 'contact':
        # The input for contact scrub is the output of the account scrub
        input_filename = f"{args.filename}_OUTPUT"
        scrubber = ContactScrubber(config, filename=input_filename)
    
    scrubber.run()

    end_time = time.time()
    print(f"\nFull Process Completed! Total Time: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()