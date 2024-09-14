import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDA:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def data_summary(self):
        """Summarize data by calculating descriptive statistics."""
        return self.data.describe()

    def data_structure(self):
        """Check the structure of the dataset."""
        return self.data.dtypes

    def check_missing_values(self):
        """Check for missing values and display them in percentage and tabular form."""
        # Calculate total missing values
        missing_values = self.data.isnull().sum()
        
        # Calculate percentage of missing values
        missing_percentage = (missing_values / len(self.data)) * 100
        
        # Create a DataFrame to display both the count and percentage of missing values
        missing_data = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percentage
        })
        
        # Filter columns with missing values greater than 0
        missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Percentage', ascending=False)
        
        return missing_data

    def univariate_analysis(self, column: str):
        """Perform univariate analysis on a given column."""
        if self.data[column].dtype == 'object':
            return self.data[column].value_counts().plot(kind='bar')
        else:
            return self.data[column].hist()

    def bivariate_analysis(self, col1: str, col2: str, hue: str = 'PostalCode'):
        """
        Perform bivariate analysis between two columns (e.g., TotalPremium and TotalClaims), 
        with an optional categorical column (e.g., PostalCode) to color or facet the scatter plot.
        """
        # Calculate monthly changes for col1 and col2
        self.data['MonthlyChange_' + col1] = self.data.groupby('TransactionMonth')[col1].diff()
        self.data['MonthlyChange_' + col2] = self.data.groupby('TransactionMonth')[col2].diff()
        
        # Plotting the scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=self.data,  # Pass the DataFrame here
            x='MonthlyChange_' + col1, 
            y='MonthlyChange_' + col2, 
            hue=hue,  # Coloring by PostalCode or any other categorical column
            palette='coolwarm'
        )
        plt.title(f"Monthly Changes in {col1} vs {col2} by {hue}")
        plt.xlabel(f"Monthly Change in {col1}")
        plt.ylabel(f"Monthly Change in {col2}")
        plt.show()


    def correlation_matrix(self):
        """
        Generate a correlation matrix for numerical features only.
        """
        # Select only the numerical columns
        numeric_data = self.data.select_dtypes(include=['number'])
        return numeric_data.corr()
    
    def plot_correlation_matrix(self):
        """
        Plot the correlation matrix using a heatmap.
        """
        corr_matrix = self.correlation_matrix()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def outlier_detection(self, column: str):
        """Detect outliers in a given numerical column."""
        sns.boxplot(x=self.data[column])
        plt.show()

    def visualize_data(self):
        """Produce creative and insightful visualizations."""
        self.data['TotalPremium'].hist(bins=50)
        plt.title('Distribution of TotalPremium')
        plt.xlabel('TotalPremium')
        plt.ylabel('Frequency')
        plt.show()

        self.data['TotalClaims'].hist(bins=50)
        plt.title('Distribution of TotalClaims')
        plt.xlabel('TotalClaims')
        plt.ylabel('Frequency')
        plt.show()

        self.plot_correlation_matrix()

    # Additional functions

    def parse_dates(self):
        """Convert date columns to datetime and extract useful features."""
        self.data['TransactionMonth'] = pd.to_datetime(self.data['TransactionMonth'], errors='coerce')
        self.data['TransactionYear'] = self.data['TransactionMonth'].dt.year
        self.data['TransactionMonthOnly'] = self.data['TransactionMonth'].dt.month

    def encode_categorical(self):
        """Encode categorical features."""
        categorical_cols = ['LegalType', 'VehicleType', 'CoverType', 'make', 'Model']
        self.data = pd.get_dummies(self.data, columns=categorical_cols)

    def premium_claim_analysis(self):
        """Create new features for premium and claims analysis."""
        self.data['ClaimToPremiumRatio'] = self.data['TotalClaims'] / self.data['TotalPremium']
        return self.data[['TotalPremium', 'TotalClaims', 'ClaimToPremiumRatio']].describe()

    def group_by_analysis(self, group_column):
        """Group data by a specific column and calculate mean statistics for numeric columns."""
        # Check if the column exists or was one-hot encoded
        original_col = group_column
        if group_column not in self.data.columns:
            # Check for one-hot encoded versions of the column
            encoded_cols = [col for col in self.data.columns if col.startswith(group_column + '_')]
            if len(encoded_cols) == 0:
                raise KeyError(f"Column '{original_col}' not found in the dataset, and no one-hot encoded version is available.")
            else:
                group_column = encoded_cols  # Use the one-hot encoded columns for grouping

        # Select numeric columns only
        numeric_cols = self.data.select_dtypes(include='number').columns

        # Perform group by operation
        try:
            if isinstance(group_column, list):  # Handling one-hot encoded columns
                return self.data.groupby(group_column)[numeric_cols].mean()
            else:
                return self.data.groupby(group_column)[numeric_cols].mean()
        except KeyError as e:
            raise KeyError(f"Grouping or mean calculation failed. Error: {str(e)}")


    def sum_insured_analysis(self):
        """Analyze SumInsured and CustomValueEstimate correlation with TotalPremium."""
        return self.data[['SumInsured', 'CustomValueEstimate', 'TotalPremium']].corr()

    def filter_outliers(self, column: str):
        """Filter out outliers based on a column using the IQR method."""
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]

    def geographic_analysis(self):
        """Analyze premium trends by geographic columns."""
        return self.data.groupby('Province')[['TotalPremium', 'TotalClaims']].mean().plot(kind='bar')

