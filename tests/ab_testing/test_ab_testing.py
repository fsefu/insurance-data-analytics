import sys
import unittest
import pandas as pd
from scipy.stats import ttest_ind
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from ab_testing.data_processor import DataProcessor, HypothesisTester

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        """Set up test data for the unit tests."""
        self.data = pd.DataFrame({
            'Province': ['Gauteng', 'Western Cape', 'Gauteng', 'Western Cape'],
            'ZipCode': ['1000', '2000', '1000', '2000'],
            'Gender': ['Male', 'Female', 'Male', 'Female'],
            'TotalClaims': [2000, 1500, 2100, 1600],
            'TotalPremium': [3000, 2500, 3200, 2600]
        })
        self.data_processor = DataProcessor(self.data)

    def test_select_kpi(self):
        """Test selecting a valid KPI column."""
        kpi = self.data_processor.select_kpi('TotalClaims')
        pd.testing.assert_series_equal(kpi, self.data['TotalClaims'])
        
        with self.assertRaises(ValueError):
            self.data_processor.select_kpi('InvalidKPI')

    def test_segment_data(self):
        """Test segmenting the data into two valid groups."""
        group_a, group_b = self.data_processor.segment_data(
            'Province',
            self.data['Province'] == 'Gauteng',
            self.data['Province'] == 'Western Cape'
        )
        self.assertEqual(len(group_a), 2)
        self.assertEqual(len(group_b), 2)
        
        with self.assertRaises(ValueError):
            self.data_processor.segment_data(
                'Province',
                self.data['Province'] == 'Invalid',
                self.data['Province'] == 'Western Cape'
            )


class TestHypothesisTester(unittest.TestCase):

    def setUp(self):
        """Set up test data for hypothesis testing."""
        self.data = pd.DataFrame({
            'Province': ['Gauteng', 'Western Cape', 'Gauteng', 'Western Cape'],
            'TotalClaims': [2000, 1500, 2100, 1600],
            'TotalPremium': [3000, 2500, 3200, 2600]
        })
        self.hypothesis_tester = HypothesisTester()
        self.group_a = self.data[self.data['Province'] == 'Gauteng']
        self.group_b = self.data[self.data['Province'] == 'Western Cape']

    def test_t_test(self):
        """Test the t-test method for numerical columns."""
        p_value = self.hypothesis_tester.t_test(self.group_a, self.group_b, 'TotalClaims')
        stat, expected_p_value = ttest_ind(self.group_a['TotalClaims'], self.group_b['TotalClaims'], equal_var=False)
        self.assertAlmostEqual(p_value, expected_p_value)

    def test_analyze_results(self):
        """Test analyzing p-value results to determine hypothesis rejection."""
        result = self.hypothesis_tester.analyze_results(0.01)
        self.assertEqual(result, "Reject the null hypothesis")

        result = self.hypothesis_tester.analyze_results(0.1)
        self.assertEqual(result, "Fail to reject the null hypothesis")

    def test_add_result(self):
        """Test adding a result to the hypothesis tester."""
        self.hypothesis_tester.add_result('Test Hypothesis', 0.05, "Reject the null hypothesis")
        self.assertEqual(len(self.hypothesis_tester.results), 1)
        self.assertEqual(self.hypothesis_tester.results[0]['Hypothesis'], 'Test Hypothesis')


if __name__ == '__main__':
    unittest.main()
