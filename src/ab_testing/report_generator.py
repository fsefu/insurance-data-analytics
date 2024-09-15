import pandas as pd

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
