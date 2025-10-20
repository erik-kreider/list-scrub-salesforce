import argparse
import configparser
from src.datascrubber.scrubbing import AccountScrubber, ContactScrubber

def main():
    """
    Main entry point for the data scrubbing application.
    """
    # Load configuration
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Set up command-line parser
    parser = argparse.ArgumentParser(description="Scrub lists against static Salesforce data.")
    subparsers = parser.add_subparsers(dest='scrub_type', required=True)

    # Account Scrubber command
    parser_account = subparsers.add_parser('account', help="Run the account scrub.")
    parser_account.add_argument('filename', type=str, help="The Excel file to scrub (without .xlsx).")

    # Contact Scrubber command
    parser_contact = subparsers.add_parser('contact', help="Run the contact scrub.")
    parser_contact.add_argument('filename', type=str, help="The account scrub OUTPUT file to use.")

    args = parser.parse_args()

    # Instantiate and run the appropriate scrubber
    if args.scrub_type == 'account':
        scrubber = AccountScrubber(config, args.filename)
    elif args.scrub_type == 'contact':
        scrubber = ContactScrubber(config, args.filename)
    
    scrubber.run()

if __name__ == '__main__':
    main()