import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

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
        """
        group_a = self.data[group_a_condition]
        group_b = self.data[group_b_condition]
        
        if group_a.empty or group_b.empty:
            raise ValueError(f"Segmentation resulted in empty group(s) for feature '{feature}'.")
        return group_a, group_b

class HypothesisTester:
    """Handles hypothesis testing using statistical methods."""
    
    def __init__(self):
        self.results = []
    
    def chi_squared_test(self, group_a, group_b, feature):
        """Performs a chi-squared test for categorical features."""
        contingency_table = pd.crosstab(group_a[feature], group_b[feature])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        return p_value
    
    def t_test(self, group_a, group_b, kpi):
        """Performs a t-test for numerical features like claims or premium."""
        stat, p_value = ttest_ind(group_a[kpi], group_b[kpi], equal_var=False)
        return p_value
    
    def analyze_results(self, p_value, alpha=0.05):
        """Analyzes the p-value to accept or reject the null hypothesis."""
        if p_value < alpha:
            return "Reject the null hypothesis"
        return "Fail to reject the null hypothesis"
    
    def add_result(self, hypothesis_name, p_value, result):
        """Stores the test results."""
        self.results.append({
            'Hypothesis': hypothesis_name,
            'P-Value': p_value,
            'Result': result
        })

class ReportGenerator:
    """Generates a report from the hypothesis test results."""
    
    def __init__(self, results):
        self.results = results
    
    def generate_report(self):
        """Compiles the results into a report format."""
        report = "\nA/B Hypothesis Testing Report\n"
        report += "="*40 + "\n"
        
        for result in self.results:
            report += f"Hypothesis: {result['Hypothesis']}\n"
            report += f"P-Value: {result['P-Value']}\n"
            report += f"Result: {result['Result']}\n"
            report += "-"*40 + "\n"
        
        return report

# # Example Usage
# if __name__ == "__main__":
#     # Example data loading (replace with actual dataset path)
#     df = pd.read_csv('insurance_data.csv')

#     # Instantiate classes
#     data_processor = DataProcessor(df)
#     hypothesis_tester = HypothesisTester()

#     # Null Hypothesis 1: Risk difference across provinces
#     group_a_prov, group_b_prov = data_processor.segment_data(
#         'Province', 
#         df['Province'] == 'Gauteng', 
#         df['Province'] == 'Western Cape'
#     )
#     p_value_prov = hypothesis_tester.t_test(group_a_prov, group_b_prov, 'TotalClaims')
#     result_prov = hypothesis_tester.analyze_results(p_value_prov)
#     hypothesis_tester.add_result('Risk difference across provinces', p_value_prov, result_prov)

#     # Null Hypothesis 2: Risk difference between zip codes
#     group_a_zip, group_b_zip = data_processor.segment_data(
#         'ZipCode', 
#         df['ZipCode'] == '1000', 
#         df['ZipCode'] == '2000'
#     )
#     p_value_zip = hypothesis_tester.t_test(group_a_zip, group_b_zip, 'TotalClaims')
#     result_zip = hypothesis_tester.analyze_results(p_value_zip)
#     hypothesis_tester.add_result('Risk difference across zip codes', p_value_zip, result_zip)

#     # Null Hypothesis 3: Margin difference between zip codes
#     group_a_margin, group_b_margin = data_processor.segment_data(
#         'ZipCode', 
#         df['ZipCode'] == '1000', 
#         df['ZipCode'] == '2000'
#     )
#     # Assuming we have a 'Margin' column: Margin = TotalPremium - TotalClaims
#     df['Margin'] = df['TotalPremium'] - df['TotalClaims']
#     p_value_margin = hypothesis_tester.t_test(group_a_margin, group_b_margin, 'Margin')
#     result_margin = hypothesis_tester.analyze_results(p_value_margin)
#     hypothesis_tester.add_result('Margin difference between zip codes', p_value_margin, result_margin)

#     # Null Hypothesis 4: Risk difference between men and women
#     group_a_gender, group_b_gender = data_processor.segment_data(
#         'Gender', 
#         df['Gender'] == 'Male', 
#         df['Gender'] == 'Female'
#     )
#     p_value_gender = hypothesis_tester.chi_squared_test(group_a_gender, group_b_gender, 'TotalClaims')
#     result_gender = hypothesis_tester.analyze_results(p_value_gender)
#     hypothesis_tester.add_result('Risk difference between men and women', p_value_gender, result_gender)

#     # Generate report
#     report_generator = ReportGenerator(hypothesis_tester.results)
#     print(report_generator.generate_report())
