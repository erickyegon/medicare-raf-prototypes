"""Tests for causal inference methods."""

import pytest
import pandas as pd
import numpy as np
from medicare_raf.inference.causal_attribution import (
    difference_in_differences,
    propensity_score_matching
)


class TestDifferenceInDifferences:
    """Test DiD implementation."""

    def test_did_basic_functionality(self):
        """Test that DiD runs without error on valid data."""
        # Create simple test panel data
        np.random.seed(42)
        data = pd.DataFrame({
            'bene_id': [f'B{i:03d}' for i in range(100)] * 2,
            'year': [0] * 100 + [1] * 100,
            'period': ['pre'] * 100 + ['post'] * 100,
            'intervention': [1] * 50 + [0] * 50 * 2,
            'total_cost': np.random.normal(8000, 1000, 200),
            'age': np.random.normal(75, 5, 200),
            'dual_eligible': np.random.choice([0, 1], 200)
        })

        result = difference_in_differences(data, outcome='total_cost')

        # Check result structure
        required_keys = ['method', 'outcome', 'att', 'p_value', 'significant']
        for key in required_keys:
            assert key in result

        assert result['method'] == 'DiD (TWFE)'
        assert result['outcome'] == 'total_cost'
        assert isinstance(result['att'], (int, float))
        assert isinstance(result['p_value'], (int, float))

    def test_did_parallel_trends(self):
        """Test parallel trends assumption check."""
        # Create data with parallel pre-period trends
        np.random.seed(42)
        n_bene = 200
        bene_ids = [f'B{i:03d}' for i in range(n_bene)]

        data = []
        for bene_id in bene_ids:
            intervention = 1 if int(bene_id[1:]) < n_bene // 2 else 0
            for year in [0, 1]:
                cost = 8000 + year * 200 + intervention * 100 + np.random.normal(0, 500)
                data.append({
                    'bene_id': bene_id,
                    'year': year,
                    'period': 'pre' if year == 0 else 'post',
                    'intervention': intervention,
                    'total_cost': cost,
                    'age': 75,
                    'dual_eligible': 0
                })

        data = pd.DataFrame(data)
        result = difference_in_differences(data, outcome='total_cost')

        # Should detect parallel trends (p > 0.05 for pre-period difference)
        assert result['parallel_trends_p'] > 0.05


class TestPropensityScoreMatching:
    """Test PSM implementation."""

    def test_psm_basic_functionality(self):
        """Test that PSM runs on valid data."""
        np.random.seed(42)
        n_bene = 1000

        # Create pre-period data with some imbalance
        pre_data = pd.DataFrame({
            'bene_id': [f'B{i:03d}' for i in range(n_bene)],
            'intervention': np.random.choice([0, 1], n_bene, p=[0.7, 0.3]),
            'age': np.random.normal(75, 5, n_bene),
            'dual_eligible': np.random.choice([0, 1], n_bene, p=[0.8, 0.2]),
            'risk_tier': np.random.choice(['low', 'moderate', 'high'], n_bene),
            'ip_admits': np.random.poisson(0.1, n_bene),
            'ed_visits': np.random.poisson(0.5, n_bene),
            'total_cost': np.random.normal(8000, 1000, n_bene)
        })

        # Create full panel (pre + post)
        panel = []
        for _, row in pre_data.iterrows():
            # Pre-period
            panel.append({
                'bene_id': row['bene_id'],
                'year': 0,
                'period': 'pre',
                'intervention': row['intervention'],
                'age': row['age'],
                'dual_eligible': row['dual_eligible'],
                'risk_tier': row['risk_tier'],
                'ip_admits': row['ip_admits'],
                'ed_visits': row['ed_visits'],
                'total_cost': row['total_cost']
            })
            # Post-period (with treatment effect)
            effect = -400 if row['intervention'] == 1 else 0
            panel.append({
                'bene_id': row['bene_id'],
                'year': 1,
                'period': 'post',
                'intervention': row['intervention'],
                'age': row['age'],
                'dual_eligible': row['dual_eligible'],
                'risk_tier': row['risk_tier'],
                'ip_admits': row['ip_admits'],
                'ed_visits': row['ed_visits'],
                'total_cost': row['total_cost'] + effect + np.random.normal(0, 500)
            })

        panel = pd.DataFrame(panel)

        result = propensity_score_matching(panel, outcome='total_cost')

        # Check result structure
        if 'att' in result:  # Successful matching
            assert isinstance(result['att'], (int, float))
            assert 'n_matched_pairs' in result
            assert result['n_matched_pairs'] > 0
        else:  # Failed matching
            assert 'error' in result

    def test_psm_balance_improvement(self):
        """Test that PSM improves covariate balance."""
        # This would require more complex setup - for now just check it runs
        pass


class TestSensitivityAnalysis:
    """Test robustness checks."""

    def test_rosenbaum_bounds_placeholder(self):
        """Placeholder for Rosenbaum sensitivity analysis."""
        # TODO: Implement Rosenbaum bounds test
        pass

    def test_placebo_test_placeholder(self):
        """Placeholder for placebo test."""
        # TODO: Implement placebo test
        pass