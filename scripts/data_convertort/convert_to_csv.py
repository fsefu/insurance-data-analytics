import csv
import os

class TxtToCSVConverter:
    def __init__(self, input_file: str, output_file: str):
        """
        Initialize the converter with the input and output file paths.

        :param input_file: Path to the pipe-separated input file.
        :param output_file: Path to the output CSV file.
        """
        self.input_file = input_file
        self.output_file = output_file

    def convert(self):
        """
        Convert the pipe-separated file to a CSV file if the CSV file does not already exist.
        """
        if os.path.exists(self.output_file):
            print(f"Output file already exists: {self.output_file}. Conversion skipped.")
            return

        try:
            with open(self.input_file, 'r') as infile, open(self.output_file, 'w', newline='') as outfile:
                reader = csv.reader(infile, delimiter='|')
                writer = csv.writer(outfile)
                
                # Write the content to the CSV file
                for row in reader:
                    writer.writerow(row)
            print(f"Conversion successful: {self.output_file}")
        except Exception as e:
            print(f"An error occurred during conversion: {e}")
