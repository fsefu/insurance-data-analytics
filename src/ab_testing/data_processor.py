import pandas as pd

class DataProcessor:
    """Responsible for preparing and segmenting the data."""
    
    def __init__(self, data):
        self.data = data
    
    def select_kpi(self, kpi_col):
        """Select the KPI for the A/B test (e.g., TotalClaims, TotalPremium)."""
        if kpi_col not in self.data.columns:
            raise ValueError(f"KPI column '{kpi_col}' does not exist in the dataset.")
        return self.data[kpi_col]
    
    def segment_data(self, feature, group_a_condition, group_b_condition):
        """
        Segments the data into Group A and Group B based on conditions.
        
        Parameters:
        feature: str -> The column used to create segments (e.g., province, gender, etc.)
        group_a_condition: condition -> Condition for Group A (e.g., data['province'] == 'A')
        group_b_condition: condition -> Condition for Group B (e.g., data['province'] == 'B')
        
        Returns:
        DataFrames for Group A and Group B
        """
        group_a = self.data[group_a_condition]
        group_b = self.data[group_b_condition]
        
        if group_a.empty or group_b.empty:
            raise ValueError(f"Segmentation resulted in empty group(s) for feature '{feature}'.")

        return group_a, group_b
    
    def get_testable_features(self):
        """Returns a list of all available features for testing."""
        return self.data.columns


