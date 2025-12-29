"""
Unit Tests for Statistical Analysis Module
Tests all functions in statistical_analysis.py with output suppression
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import sys
import io
import os
from pathlib import Path

# Suppress Unicode encoding errors by setting UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from statistical_analysis import (
    perform_friedman_test,
    calculate_kendalls_w,
    perform_nemenyi_posthoc,
    apply_hommel_correction,
    calculate_cliffs_delta,
    calculate_pairwise_effect_sizes,
    save_results_to_csv,
    run_complete_statistical_analysis
)


class TestFriedmanTest(unittest.TestCase):
    """Test Friedman ANOVA test function"""
    
    def setUp(self):
        """Set up test data"""
        self.model_names = ['RandomForest', 'SVM', 'XGBoost']
        
        # Different test matrices for different scenarios
        # Pattern 1: Clear separation between models
        self.data_clear_diff = np.array([
            [0.73, 0.74, 0.75],
            [0.72, 0.73, 0.74],
            [0.71, 0.72, 0.73],
            [0.73, 0.74, 0.75],
            [0.72, 0.73, 0.74],
            [0.71, 0.72, 0.73],
            [0.73, 0.74, 0.75],
            [0.72, 0.73, 0.74],
            [0.71, 0.72, 0.73],
            [0.73, 0.74, 0.75],
            [0.72, 0.73, 0.74],
            [0.71, 0.72, 0.73]
        ])
        
        # Pattern 2: All identical values (should have high p-value)
        self.data_identical = np.full((12, 3), 0.75)
        
        # Pattern 3: Random values (may or may not be significant)
        np.random.seed(42)
        self.data_random = np.random.uniform(0.70, 0.76, (12, 3))
        
        # Pattern 4: Large differences (should have very low p-value)
        self.data_large_diff = np.array([
            [0.60, 0.70, 0.80],
            [0.61, 0.71, 0.81],
            [0.62, 0.72, 0.82],
            [0.60, 0.70, 0.80],
            [0.61, 0.71, 0.81],
            [0.62, 0.72, 0.82],
            [0.60, 0.70, 0.80],
            [0.61, 0.71, 0.81],
            [0.62, 0.72, 0.82],
            [0.60, 0.70, 0.80],
            [0.61, 0.71, 0.81],
            [0.62, 0.72, 0.82]
        ])
    
    def test_friedman_output_structure(self):
        """Test that Friedman test returns correct structure"""
        result = perform_friedman_test(self.data_clear_diff, self.model_names)
        
        self.assertIsInstance(result, dict)
        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('is_significant', result)
        self.assertIn('df', result)
    
    def test_friedman_degrees_of_freedom(self):
        """Test that degrees of freedom is correct for different data"""
        # Test with different matrices
        for data in [self.data_clear_diff, self.data_random, self.data_large_diff]:
            result = perform_friedman_test(data, self.model_names)
            expected_df = len(self.model_names) - 1
            self.assertEqual(result['df'], expected_df)
    
    def test_friedman_statistic_positive(self):
        """Test that chi-square statistic is positive for all data"""
        for data in [self.data_clear_diff, self.data_identical, self.data_random, self.data_large_diff]:
            result = perform_friedman_test(data, self.model_names)
            # Handle NaN case (identical values may produce NaN statistic)
            if not np.isnan(result['statistic']):
                self.assertGreaterEqual(result['statistic'], 0)
    
    def test_friedman_pvalue_range(self):
        """Test that p-value is between 0 and 1 for various datasets"""
        for data in [self.data_clear_diff, self.data_identical, self.data_random, self.data_large_diff]:
            result = perform_friedman_test(data, self.model_names)
            # Handle NaN case (identical values may produce NaN p-value)
            if not np.isnan(result['p_value']):
                self.assertGreaterEqual(result['p_value'], 0)
                self.assertLessEqual(result['p_value'], 1)
    
    def test_friedman_identical_values_not_significant(self):
        """Test that identical values produce non-significant result"""
        result = perform_friedman_test(self.data_identical, self.model_names)
        # With identical values, p-value should be high (not significant)
        # Note: May produce NaN, which should be handled as not significant
        if not np.isnan(result['p_value']):
            self.assertGreater(result['p_value'], 0.05)
    
    def test_friedman_large_diff_significant(self):
        """Test that large differences produce significant result"""
        result = perform_friedman_test(self.data_large_diff, self.model_names)
        # With large consistent differences, should be significant
        self.assertLess(result['p_value'], 0.05)
        self.assertTrue(result['is_significant'])
    
    def test_friedman_different_pvalues(self):
        """Test that different data patterns produce different p-values"""
        result_clear = perform_friedman_test(self.data_clear_diff, self.model_names)
        result_large = perform_friedman_test(self.data_large_diff, self.model_names)
        result_random = perform_friedman_test(self.data_random, self.model_names)
        
        # Collect valid p-values (not NaN)
        pvalues = []
        for r in [result_clear, result_large, result_random]:
            if not np.isnan(r['p_value']):
                pvalues.append(r['p_value'])
        
        # Should have at least 2 different p-values
        self.assertGreater(len(set(pvalues)), 1, 
                          f"All p-values are the same: {pvalues}")


class TestKendallsW(unittest.TestCase):
    """Test Kendall's W effect size calculation"""
    
    def test_kendalls_w_output_structure(self):
        """Test that Kendall's W returns correct structure"""
        result = calculate_kendalls_w(friedman_stat=10.5, n_blocks=12, n_models=3)
        
        self.assertIsInstance(result, dict)
        self.assertIn('kendalls_w', result)
        self.assertIn('interpretation', result)
    
    def test_kendalls_w_calculation(self):
        """Test that Kendall's W is calculated correctly"""
        result = calculate_kendalls_w(friedman_stat=10.5, n_blocks=12, n_models=3)
        
        expected_w = 10.5 / (12 * 2)
        self.assertAlmostEqual(result['kendalls_w'], expected_w, places=6)
    
    def test_kendalls_w_range(self):
        """Test that Kendall's W is between 0 and 1"""
        result = calculate_kendalls_w(friedman_stat=20.0, n_blocks=12, n_models=3)
        
        self.assertGreaterEqual(result['kendalls_w'], 0)
        self.assertLessEqual(result['kendalls_w'], 1)


class TestNemenyiPosthoc(unittest.TestCase):
    """Test Nemenyi post-hoc test"""
    
    def setUp(self):
        """Set up test data"""
        self.data_matrix = np.array([
            [0.73, 0.74, 0.75],
            [0.72, 0.73, 0.74],
            [0.71, 0.72, 0.73],
            [0.73, 0.74, 0.75],
            [0.72, 0.73, 0.74],
            [0.71, 0.72, 0.73],
            [0.73, 0.74, 0.75],
            [0.72, 0.73, 0.74],
            [0.71, 0.72, 0.73],
            [0.73, 0.74, 0.75],
            [0.72, 0.73, 0.74],
            [0.71, 0.72, 0.73]
        ])
        self.model_names = ['RandomForest', 'SVM', 'XGBoost']
    
    def test_nemenyi_output_structure(self):
        """Test that Nemenyi test returns correct structure"""
        result = perform_nemenyi_posthoc(self.data_matrix, self.model_names)
        
        self.assertIsInstance(result, dict)
        self.assertIn('p_matrix', result)
        self.assertIn('pairwise_results', result)
    
    def test_nemenyi_pmatrix_shape(self):
        """Test that p-value matrix has correct shape"""
        result = perform_nemenyi_posthoc(self.data_matrix, self.model_names)
        self.assertEqual(result['p_matrix'].shape, (3, 3))
    
    def test_nemenyi_pairwise_count(self):
        """Test that correct number of pairwise comparisons"""
        result = perform_nemenyi_posthoc(self.data_matrix, self.model_names)
        self.assertEqual(len(result['pairwise_results']), 3)


class TestHommelCorrection(unittest.TestCase):
    """Test Hommel multiple comparison correction"""
    
    def setUp(self):
        """Set up test data"""
        self.posthoc_matrix = pd.DataFrame(
            [[1.0, 0.05, 0.01],
             [0.05, 1.0, 0.03],
             [0.01, 0.03, 1.0]],
            columns=['RandomForest', 'SVM', 'XGBoost'],
            index=['RandomForest', 'SVM', 'XGBoost']
        )
        self.model_names = ['RandomForest', 'SVM', 'XGBoost']
    
    def test_hommel_output_structure(self):
        """Test that Hommel correction returns correct structure"""
        result = apply_hommel_correction(self.posthoc_matrix, self.model_names)
        
        self.assertIsInstance(result, dict)
        self.assertIn('comparisons', result)
        self.assertIn('n_significant', result)
        self.assertIn('n_comparisons', result)
    
    def test_hommel_comparison_count(self):
        """Test that correct number of comparisons"""
        result = apply_hommel_correction(self.posthoc_matrix, self.model_names)
        self.assertEqual(result['n_comparisons'], 3)


class TestCliffsDelta(unittest.TestCase):
    """Test Cliff's Delta effect size calculation"""
    
    def test_cliffs_delta_identical_groups(self):
        """Test Cliff's Delta for identical groups"""
        group1 = np.array([0.7, 0.7, 0.7, 0.7])
        group2 = np.array([0.7, 0.7, 0.7, 0.7])
        
        delta, interpretation = calculate_cliffs_delta(group1, group2)
        
        self.assertEqual(delta, 0.0)
        self.assertEqual(interpretation, "Negligible")
    
    def test_cliffs_delta_group1_better(self):
        """Test Cliff's Delta when group1 is better"""
        group1 = np.array([0.8, 0.8, 0.8, 0.8])
        group2 = np.array([0.7, 0.7, 0.7, 0.7])
        
        delta, interpretation = calculate_cliffs_delta(group1, group2)
        self.assertGreater(delta, 0)
    
    def test_cliffs_delta_range(self):
        """Test that Cliff's Delta is between -1 and 1"""
        group1 = np.random.uniform(0.6, 0.8, 10)
        group2 = np.random.uniform(0.6, 0.8, 10)
        
        delta, interpretation = calculate_cliffs_delta(group1, group2)
        
        self.assertGreaterEqual(delta, -1)
        self.assertLessEqual(delta, 1)


class TestPairwiseEffectSizes(unittest.TestCase):
    """Test pairwise effect sizes calculation"""
    
    def setUp(self):
        """Set up test data"""
        self.data_matrix = np.array([
            [0.73, 0.74, 0.75],
            [0.72, 0.73, 0.74],
            [0.71, 0.72, 0.73],
            [0.73, 0.74, 0.75],
            [0.72, 0.73, 0.74],
            [0.71, 0.72, 0.73],
            [0.73, 0.74, 0.75],
            [0.72, 0.73, 0.74],
            [0.71, 0.72, 0.73],
            [0.73, 0.74, 0.75],
            [0.72, 0.73, 0.74],
            [0.71, 0.72, 0.73]
        ])
        self.model_names = ['RandomForest', 'SVM', 'XGBoost']
    
    def test_pairwise_effect_sizes_count(self):
        """Test that correct number of pairwise comparisons"""
        result = calculate_pairwise_effect_sizes(self.data_matrix, self.model_names)
        self.assertEqual(len(result), 3)
    
    def test_pairwise_effect_sizes_structure(self):
        """Test that each result has correct structure"""
        result = calculate_pairwise_effect_sizes(self.data_matrix, self.model_names)
        
        for pair in result:
            self.assertIn('Model_1', pair)
            self.assertIn('Model_2', pair)
            self.assertIn('Cliffs_Delta', pair)
            self.assertIn('Interpretation', pair)


class TestSaveResults(unittest.TestCase):
    """Test saving results to CSV"""
    
    def setUp(self):
        """Set up test data and temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        
        self.all_results = {
            'data_matrix': np.array([[0.73, 0.74, 0.75], [0.72, 0.73, 0.74], [0.71, 0.72, 0.73]]),
            'block_labels': ['Block1', 'Block2', 'Block3'],
            'model_names': ['RandomForest', 'SVM', 'XGBoost'],
            'friedman': {
                'statistic': 10.5,
                'p_value': 0.005,
                'df': 2,
                'is_significant': True
            },
            'effect_size': {
                'kendalls_w': 0.45,
                'interpretation': 'Medium effect'
            },
            'posthoc': {
                'pairwise_results': [
                    {'Model_1': 'RF', 'Model_2': 'SVM', 'p_value': 0.05, 'significant': True}
                ]
            },
            'hommel': {
                'comparisons': [
                    {'Model_1': 'RF', 'Model_2': 'SVM', 'original_p': 0.05,
                     'corrected_p': 0.075, 'significant': False}
                ],
                'n_significant': 0,
                'n_comparisons': 3
            },
            'pairwise_effect_sizes': [
                {'Model_1': 'RF', 'Model_2': 'SVM', 'Cliffs_Delta': -0.2,
                 'Interpretation': 'Small', 'Favors': 'SVM'}
            ]
        }
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_save_results_creates_directory(self):
        """Test that output directory is created"""
        output_dir = Path(self.temp_dir) / 'test_results'
        save_results_to_csv(self.all_results, output_dir=str(output_dir))
        
        self.assertTrue(output_dir.exists())
    
    def test_save_results_creates_files(self):
        """Test that all expected CSV files are created"""
        output_dir = Path(self.temp_dir) / 'test_results'
        save_results_to_csv(self.all_results, output_dir=str(output_dir))
        
        expected_files = [
            'data_matrix.csv',
            'friedman_test_results.csv',
            'effect_size_results.csv'
        ]
        
        for filename in expected_files:
            file_path = output_dir / filename
            self.assertTrue(file_path.exists(), f"{filename} was not created")


class TestCompleteStatisticalAnalysis(unittest.TestCase):
    """Test complete statistical analysis workflow"""
    
    def setUp(self):
        """Set up test data"""
        self.data_matrix = np.array([
            [0.73, 0.74, 0.75], [0.72, 0.73, 0.74], [0.71, 0.72, 0.73],
            [0.73, 0.74, 0.75], [0.72, 0.73, 0.74], [0.71, 0.72, 0.73],
            [0.73, 0.74, 0.75], [0.72, 0.73, 0.74], [0.71, 0.72, 0.73],
            [0.73, 0.74, 0.75], [0.72, 0.73, 0.74], [0.71, 0.72, 0.73]
        ])
        self.model_names = ['RandomForest', 'SVM', 'XGBoost']
        self.block_labels = [f'Block{i+1}' for i in range(12)]
    
    def test_accepts_numpy_array(self):
        """Test that function accepts numpy array"""
        result = run_complete_statistical_analysis(
            data_matrix=self.data_matrix,
            model_names=self.model_names,
            block_labels=self.block_labels,
            save_to_csv=False
        )
        
        self.assertIsInstance(result, dict)
    
    def test_accepts_dataframe(self):
        """Test that function accepts DataFrame"""
        df = pd.DataFrame(
            self.data_matrix,
            columns=self.model_names,
            index=self.block_labels
        )
        
        result = run_complete_statistical_analysis(
            data_matrix=df,
            save_to_csv=False
        )
        
        self.assertIsInstance(result, dict)
    
    def test_validates_column_count(self):
        """Test that function validates 3 columns"""
        bad_matrix = np.random.rand(12, 4)
        
        with self.assertRaises(ValueError):
            run_complete_statistical_analysis(
                data_matrix=bad_matrix,
                save_to_csv=False
            )
    
    def test_validates_row_count(self):
        """Test that function validates 12 rows"""
        bad_matrix = np.random.rand(10, 3)
        
        with self.assertRaises(ValueError):
            run_complete_statistical_analysis(
                data_matrix=bad_matrix,
                save_to_csv=False
            )
    
    def test_output_structure(self):
        """Test that output has correct structure"""
        result = run_complete_statistical_analysis(
            data_matrix=self.data_matrix,
            model_names=self.model_names,
            block_labels=self.block_labels,
            save_to_csv=False
        )
        
        self.assertIn('data_matrix', result)
        self.assertIn('block_labels', result)
        self.assertIn('model_names', result)
        self.assertIn('friedman', result)


def run_tests():
    """Run all tests with verbose output"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestFriedmanTest))
    suite.addTests(loader.loadTestsFromTestCase(TestKendallsW))
    suite.addTests(loader.loadTestsFromTestCase(TestNemenyiPosthoc))
    suite.addTests(loader.loadTestsFromTestCase(TestHommelCorrection))
    suite.addTests(loader.loadTestsFromTestCase(TestCliffsDelta))
    suite.addTests(loader.loadTestsFromTestCase(TestPairwiseEffectSizes))
    suite.addTests(loader.loadTestsFromTestCase(TestSaveResults))
    suite.addTests(loader.loadTestsFromTestCase(TestCompleteStatisticalAnalysis))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result


if __name__ == '__main__':
    run_tests()
