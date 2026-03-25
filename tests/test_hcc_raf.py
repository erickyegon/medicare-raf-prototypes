"""Tests for HCC calculation logic."""

import pytest
import pandas as pd
import numpy as np
from src.medicare_raf.modeling.hcc_mapper import get_hcc_coefficient, get_hcc_label
from src.medicare_raf.modeling.raf_calculator import calculate_raf_batch


class TestHCCMapping:
    """Test HCC coefficient mapping and logic."""

    def test_known_hcc_coefficients(self):
        """Test that known HCC codes return expected coefficients."""
        # Test some known HCC coefficients from CMS v28
        assert get_hcc_coefficient(85) == 0.323  # CHF
        assert get_hcc_coefficient(96) == 0.421  # Atrial fibrillation
        assert get_hcc_coefficient(18) == 0.302  # Diabetes with complications

    def test_hcc_labels(self):
        """Test HCC label retrieval."""
        assert "Congestive Heart Failure" in get_hcc_label(85)
        assert "Atrial Fibrillation" in get_hcc_label(96)

    def test_invalid_hcc(self):
        """Test handling of invalid HCC codes."""
        assert get_hcc_coefficient(999) == 0.0  # Non-existent HCC


class TestRAFCalculation:
    """Test RAF score calculation."""

    def test_simple_raf_calculation(self):
        """Test RAF calculation for a simple case."""
        # Create a test beneficiary with known HCCs
        test_data = pd.DataFrame({
            'bene_id': ['TEST001'],
            'age': [75],
            'sex': ['F'],
            'dual_eligible': [0],
            'icd10_codes': [['I500', 'E119']],  # CHF + uncomplicated diabetes
            'risk_tier': ['high']
        })

        result = calculate_raf_batch(test_data)

        # Should have RAF score calculated
        assert 'raf_score' in result.columns
        assert len(result) == 1
        assert result['raf_score'].iloc[0] > 0

    def test_demographic_coefficients(self):
        """Test demographic coefficient calculation."""
        # Age-sex coefficients should be positive
        from src.medicare_raf.modeling.raf_calculator import get_demographic_coefficient

        coeff_75f = get_demographic_coefficient(75, 'F', dual=False)
        coeff_75m = get_demographic_coefficient(75, 'M', dual=False)

        assert coeff_75f > 0
        assert coeff_75m > 0
        assert coeff_75f != coeff_75m  # Should differ by sex

    @pytest.mark.parametrize("age,sex,expected_range", [
        (65, 'F', (0.3, 0.5)),
        (80, 'M', (0.4, 0.6)),
        (90, 'F', (0.5, 0.7)),
    ])
    def test_demographic_coefficient_ranges(self, age, sex, expected_range):
        """Test demographic coefficients are in expected ranges."""
        from src.medicare_raf.modeling.raf_calculator import get_demographic_coefficient

        coeff = get_demographic_coefficient(age, sex, dual=False)
        assert expected_range[0] <= coeff <= expected_range[1]


class TestGoldenDataset:
    """Test against manually calculated 'golden' dataset."""

    def test_golden_case_chf_afib(self):
        """Test a manually verified case: 76F with CHF + AFib + T2DM + CKD4."""
        # This is the example from the README
        test_data = pd.DataFrame({
            'bene_id': ['GOLDEN001'],
            'age': [76],
            'sex': ['F'],
            'dual_eligible': [0],
            'icd10_codes': [['I500', 'I501', 'I480', 'E1140', 'N184']],  # CHF + AFib + T2DM + CKD4
            'risk_tier': ['high']
        })

        result = calculate_raf_batch(test_data)

        # Expected components from README:
        # Demo (76F): 0.453
        # CHF (85): +0.323
        # AFib (96): +0.421
        # T2DM w/comp (18): +0.302
        # CKD4 (137): +0.138
        # CHF×AFib: +0.175
        # Total expected: 1.812

        raf_score = result['raf_score'].iloc[0]
        assert abs(raf_score - 1.812) < 0.01, f"Expected ~1.812, got {raf_score}"