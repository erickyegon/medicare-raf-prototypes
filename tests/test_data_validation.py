"""Tests for data validation."""

import pandas as pd
import pytest

from medicare_raf.data.data_generator import (
    BeneficiaryRecord,
    UtilizationRecord,
    validate_beneficiary_cohort,
)


class TestDataValidation:
    """Test data validation with Pydantic."""

    def test_valid_beneficiary_record(self):
        """Test validation of a valid beneficiary record."""
        data = {
            "bene_id": "BENE001",
            "age": 75,
            "sex": "F",
            "race_ethnicity": "White",
            "dual_eligible": 0,
            "risk_tier": "high",
            "icd10_codes": ["I500", "E119"],
            "intervention": 1,
            "county_fips": "21097",
            "plan_type": "HMO",
        }

        record = BeneficiaryRecord(**data)
        assert record.bene_id == "BENE001"
        assert record.age == 75

    def test_invalid_beneficiary_age(self):
        """Test validation rejects invalid age."""
        data = {
            "bene_id": "BENE001",
            "age": 30,  # Too young for Medicare
            "sex": "F",
            "race_ethnicity": "White",
            "dual_eligible": 0,
            "risk_tier": "high",
            "icd10_codes": ["I500"],
            "intervention": 1,
            "county_fips": "21097",
            "plan_type": "HMO",
        }

        with pytest.raises(ValueError):
            BeneficiaryRecord(**data)

    def test_invalid_icd10_code(self):
        """Test validation rejects malformed ICD-10 codes."""
        data = {
            "bene_id": "BENE001",
            "age": 75,
            "sex": "F",
            "race_ethnicity": "White",
            "dual_eligible": 0,
            "risk_tier": "high",
            "icd10_codes": ["INVALID"],  # Too short
            "intervention": 1,
            "county_fips": "21097",
            "plan_type": "HMO",
        }

        with pytest.raises(ValueError):
            BeneficiaryRecord(**data)

    def test_valid_utilization_record(self):
        """Test validation of a valid utilization record."""
        data = {
            "bene_id": "BENE001",
            "year": 0,
            "period": "pre",
            "intervention": 1,
            "risk_tier": "high",
            "age": 75,
            "sex": "F",
            "dual_eligible": 0,
            "county_fips": "21097",
            "total_cost": 8500.50,
            "ip_admits": 1,
            "ed_visits": 2,
        }

        record = UtilizationRecord(**data)
        assert record.total_cost == 8500.50
        assert record.ip_admits == 1

    def test_negative_cost_rejection(self):
        """Test validation rejects negative costs."""
        data = {
            "bene_id": "BENE001",
            "year": 0,
            "period": "pre",
            "intervention": 1,
            "risk_tier": "high",
            "age": 75,
            "sex": "F",
            "dual_eligible": 0,
            "county_fips": "21097",
            "total_cost": -100.0,  # Invalid
            "ip_admits": 1,
            "ed_visits": 2,
        }

        with pytest.raises(ValueError):
            UtilizationRecord(**data)

    def test_cohort_validation_integration(self):
        """Test full cohort validation."""
        cohort_data = pd.DataFrame(
            {
                "bene_id": ["BENE001", "BENE002"],
                "age": [75, 80],
                "sex": ["F", "M"],
                "race_ethnicity": ["White", "Black"],
                "dual_eligible": [0, 1],
                "risk_tier": ["high", "moderate"],
                "icd10_codes": [["I500"], ["E119"]],
                "intervention": [1, 0],
                "county_fips": ["21097", "21151"],
                "plan_type": ["HMO", "PPO"],
            }
        )

        validated = validate_beneficiary_cohort(cohort_data)
        assert len(validated) == 2
        assert validated["bene_id"].tolist() == ["BENE001", "BENE002"]
