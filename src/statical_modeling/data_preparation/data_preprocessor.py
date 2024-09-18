import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def handle_missing_data(self):
        """Handle missing data for numeric and categorical features separately."""
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        datetime_cols = self.data.select_dtypes(include=['datetime64[ns]']).columns

        # Identify columns with all missing values
        missing_all_numeric = self.data[numeric_cols].isna().all()

        # Drop numeric columns that are entirely missing
        if missing_all_numeric.any():
            cols_to_drop = numeric_cols[missing_all_numeric]
            print(f"Dropping columns with all missing values: {cols_to_drop.tolist()}")
            self.data.drop(columns=cols_to_drop, inplace=True)
            numeric_cols = numeric_cols[~missing_all_numeric]  # Update numeric_cols

        # Impute numeric columns with the mean
        if len(numeric_cols) > 0:
            imputer_numeric = SimpleImputer(strategy='mean')
            transformed_numeric = imputer_numeric.fit_transform(self.data[numeric_cols])
            self.data[numeric_cols] = pd.DataFrame(transformed_numeric, columns=numeric_cols, index=self.data.index)

        # Impute categorical columns with the most frequent value
        if len(categorical_cols) > 0:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            self.data[categorical_cols] = pd.DataFrame(
                imputer_categorical.fit_transform(self.data[categorical_cols]),
                columns=categorical_cols,
                index=self.data.index
            )

        # Handle datetime columns (forward fill)
        if len(datetime_cols) > 0:
            self.data[datetime_cols] = self.data[datetime_cols].fillna(method='ffill')

#     def handle_infinity(self):
#         """Replace infinite values with NaN and re-handle missing data."""
#         # self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
#         self.data.replace([np.isfinite(self.data).all(1)])

#         self.handle_missing_data()  # Re-handle missing data after replacing inf
# # df_new = df[np.isfinite(df).all(1)]

    def handle_infinity(self):
        """Replace infinite values with NaN in numeric columns and re-handle missing data."""
        # Apply replacement only on numeric columns
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        
        # Replace inf and -inf with NaN
        self.data[numeric_cols] = self.data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Check if any inf values remain (for debugging purposes)
        if np.isinf(self.data[numeric_cols]).any().any():
            print("Warning: Some infinity values were not replaced properly.")
        
        self.handle_missing_data()  # Re-handle missing data after replacing inf

    def cap_outliers(self):
        """Cap extreme outliers at the 99th percentile."""
        for col in self.data.select_dtypes(include=['number']).columns:
            cap_value = self.data[col].quantile(0.99)
            self.data[col] = self.data[col].apply(lambda x: min(x, cap_value))

    def feature_engineering(self):
        """Create new features that could be relevant to TotalPremium and TotalClaims."""
        # Handle division by zero: Replace zeros in 'TotalPremium' with NaN or a small number to avoid inf
        self.data['Claims_to_Premium'] = np.where(self.data['TotalPremium'] == 0, np.nan, 
                                                self.data['TotalClaims'] / self.data['TotalPremium'])
        
        # Optionally, you can also impute or handle NaN values in this new column
        # For example, replacing NaNs with 0 if necessary:
        self.data['Claims_to_Premium'].fillna(0, inplace=True)

    def encode_categorical_data(self):
        """One-hot encoding for categorical columns."""
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)

    def preprocess(self):
        """Full preprocessing pipeline including handling infinity and outliers."""
        self.handle_infinity()  # Handle inf values and missing data
        self.cap_outliers()  # Cap extreme outliers
        self.feature_engineering()
        self.encode_categorical_data()

    def split_data(self, target_column, test_size=0.2):
        """Split the data into training and testing sets."""
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=42)
