# Writing a unit test for the EDA class
import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from eda.eda import EDA

class TestEDA(unittest.TestCase):
    
    def setUp(self):
        """Set up a sample DataFrame for testing."""
        data = {
            'TotalPremium': [1000, 1500, np.nan, 2000, 2500],
            'TotalClaims': [500, 700, np.nan, 800, 1000],
            'SumInsured': [10000, 15000, 20000, np.nan, 25000],
            'CustomValueEstimate': [12000, 14000, 18000, 16000, 21000],
            'TransactionMonth': ['2024-01-01', '2024-02-01', np.nan, '2024-04-01', '2024-05-01'],
            'Province': ['Western Cape', 'Gauteng', 'KwaZulu-Natal', 'Gauteng', 'Western Cape'],
            'LegalType': ['Personal', 'Commercial', 'Personal', 'Personal', 'Commercial']
        }
        self.df = pd.DataFrame(data)
        self.eda = EDA(self.df)
    
    def test_data_summary(self):
        """Test the data_summary method."""
        summary = self.eda.data_summary()
        
        # Check that the summary has 8 rows (for 8 statistics)
        self.assertEqual(summary.shape[0], 8)
        
        # Check that the relevant columns are present
        self.assertIn('TotalPremium', summary.columns)
        self.assertIn('TotalClaims', summary.columns)

    def test_data_structure(self):
        """Test the data_structure method."""
        structure = self.eda.data_structure()
        self.assertEqual(structure['TotalPremium'], 'float64')
        self.assertEqual(structure['TransactionMonth'], 'object')
    
    def test_check_missing_values(self):
        """Test the check_missing_values method."""
        missing_data = self.eda.check_missing_values()
        self.assertEqual(missing_data.loc['TotalPremium', 'Missing Values'], 1)
        self.assertEqual(missing_data.loc['TotalPremium', 'Percentage'], 20.0)
    
    def test_parse_dates(self):
        """Test the parse_dates method."""
        self.eda.parse_dates()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.eda.data['TransactionMonth']))
        self.assertIn('TransactionYear', self.eda.data.columns)
        self.assertIn('TransactionMonthOnly', self.eda.data.columns)
    
    def test_correlation_matrix(self):
        """Test the correlation_matrix method."""
        corr_matrix = self.eda.correlation_matrix()
        self.assertIn('TotalPremium', corr_matrix.columns)
        self.assertAlmostEqual(corr_matrix.loc['TotalPremium', 'TotalClaims'], 1.0, delta=0.1)

    def test_premium_claim_analysis(self):
        """Test the premium_claim_analysis method."""
        analysis = self.eda.premium_claim_analysis()
        self.assertIn('ClaimToPremiumRatio', analysis.columns)
        self.assertGreater(analysis['ClaimToPremiumRatio'].mean(), 0)
    
    def test_filter_outliers(self):
        """Test the filter_outliers method."""
        filtered_data = self.eda.filter_outliers('TotalPremium')
        self.assertTrue((filtered_data['TotalPremium'] >= 1000).all())
        self.assertTrue((filtered_data['TotalPremium'] <= 2500).all())

    def test_encode_categorical(self):
        """Test the encode_categorical method."""
        self.eda.encode_categorical()
        self.assertIn('LegalType_Commercial', self.eda.data.columns)
        self.assertIn('LegalType_Personal', self.eda.data.columns)

if __name__ == '__main__':
    unittest.main()

