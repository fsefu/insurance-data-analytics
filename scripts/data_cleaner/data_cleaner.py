import pandas as pd

class DataCleaner:
    def __init__(self, df):
        """
        Initialize DataCleaner with the dataframe.
        
        Parameters:
        df (pd.DataFrame): The dataframe to be cleaned.
        """
        self.df = df
        self.gender_mode = self._calculate_gender_mode()

    def _calculate_gender_mode(self):
        """
        Calculate the mode of the 'Gender' column, restricted to 'Male' or 'Female'.
        Returns the most frequent gender or 'Male' if there's a tie.
        """
        gender_mode = self.df[self.df['Gender'].isin(['Male', 'Female'])]['Gender'].mode()
        return gender_mode[0] if not gender_mode.empty else 'Male'

    def clean_gender(self):
        # Mapping titles to genders
        title_gender_map = {
            'Mr': 'Male',
            'Miss': 'Female',
            'Mrs': 'Female',
            'Ms': 'Female',
            'Dr': self.gender_mode  # Use the mode of 'Gender' for Dr
        }

        def infer_gender(row):
            if row['Gender'] not in ['Male', 'Female']:
                return title_gender_map.get(row['Title'], row['Gender'])
            return row['Gender']

        # Apply the gender inference
        self.df['Gender'] = self.df.apply(infer_gender, axis=1)

        # Replace 'Male' with 1 and 'Female' with 0
        self.df['Gender'] = self.df['Gender'].map({'Male': 1, 'Female': 0})

    def add_margin_column(self):
        """
        Add a 'Margin' column, where Margin = TotalPremium - TotalClaims.
        """
        if 'TotalPremium' in self.df.columns and 'TotalClaims' in self.df.columns:
            self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
        else:
            raise KeyError("Columns 'TotalPremium' and 'TotalClaims' must be present in the dataframe.")

    def get_cleaned_data(self):
        """
        Return the cleaned dataframe.
        """
        return self.df
