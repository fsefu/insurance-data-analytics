import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, zscore

class HypothesisTester:
    """Handles hypothesis testing using statistical methods."""
    
    def __init__(self):
        self.results = []
    
    def chi_squared_test(self, group_a, group_b, feature):
        """Performs a chi-squared test for categorical features."""
        contingency_table = pd.crosstab(group_a[feature], group_b[feature])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return p_value
    
    def t_test(self, group_a, group_b, kpi):
        """Performs a t-test for numerical features like claims or premium."""
        stat, p_value = ttest_ind(group_a[kpi], group_b[kpi], equal_var=False)
        return p_value
    
    def z_test(self, group_a, group_b, kpi):
        """Performs a z-test for larger sample sizes of numerical features."""
        z_stat = (group_a[kpi].mean() - group_b[kpi].mean()) / \
                 (group_a[kpi].std() / len(group_a) + group_b[kpi].std() / len(group_b))**0.5
        return zscore(z_stat)
    
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
