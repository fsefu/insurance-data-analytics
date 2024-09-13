# eda.py
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
        """Check for missing values in the dataset."""
        return self.data.isnull().sum()

    def univariate_analysis(self, column: str):
        """Perform univariate analysis on a given column."""
        if self.data[column].dtype == 'object':
            return self.data[column].value_counts().plot(kind='bar')
        else:
            return self.data[column].hist()

    def bivariate_analysis(self, col1: str, col2: str):
        """Perform bivariate analysis between two columns."""
        return sns.scatterplot(x=self.data[col1], y=self.data[col2])

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

