"""Tests for fraud domain modules."""

import pytest
from src.domains import hope_florida, insurance, sheriff, corporate, pandemic


class TestHopeFlorida:
    """Tests for Hope Florida fund routing detection."""

    def test_generate_settlement_routing(self, seed):
        """Test settlement generation."""
        settlements = hope_florida.generate_settlement_routing(10, 0.15, seed)
        assert len(settlements) == 10
        assert all(s.settlement_id for s in settlements)

    def test_detect_foundation_pac_link(self, sample_settlement):
        """Test PAC link detection."""
        result = hope_florida.detect_foundation_pac_link(sample_settlement)
        assert "pac_linked" in result
        assert "path" in result

    def test_score_routing_risk(self, sample_settlement):
        """Test risk scoring."""
        score = hope_florida.score_routing_risk(sample_settlement)
        assert 0.0 <= score <= 1.0

    def test_detect_fraud(self, sample_settlement):
        """Test fraud detection."""
        detected, details = hope_florida.detect_fraud(sample_settlement)
        assert isinstance(detected, bool)
        assert "risk_score" in details

    def test_run_detection(self, seed):
        """Test full detection run."""
        result = hope_florida.run_detection(n_settlements=20, fraud_rate=0.15, seed=seed)
        assert "detection_rate" in result
        assert "receipts" in result
        assert len(result["receipts"]) == 20


class TestInsurance:
    """Tests for insurance influence detection."""

    def test_generate_contributions(self, seed):
        """Test contribution generation."""
        contributions = insurance.generate_insurance_contributions(5, 10, seed)
        assert len(contributions) > 0

    def test_generate_votes(self, seed):
        """Test vote generation."""
        legislators = [f"LEG-{i}" for i in range(5)]
        votes = insurance.generate_legislative_votes(3, legislators, seed)
        assert len(votes) == 15  # 3 bills * 5 legislators

    def test_detect_correlation(self, seed):
        """Test vote-contribution correlation."""
        contributions = insurance.generate_insurance_contributions(5, 10, seed)
        legislators = list(set(c.recipient_legislator for c in contributions))
        votes = insurance.generate_legislative_votes(3, legislators, seed)

        result = insurance.detect_vote_contribution_correlation(contributions, votes)
        assert "correlation" in result

    def test_run_detection(self, seed):
        """Test full detection run."""
        result = insurance.run_detection(
            n_insurers=5, n_legislators=10, fraud_rate=0.10, seed=seed
        )
        assert "influence_score" in result
        assert "receipts" in result


class TestSheriff:
    """Tests for sheriff contract detection."""

    def test_generate_contracts(self, seed):
        """Test contract generation."""
        contracts = sheriff.generate_sheriff_contracts(20, 0.08, seed)
        assert len(contracts) == 20

    def test_detect_no_show_pattern(self, sample_contract):
        """Test no-show pattern detection."""
        result = sheriff.detect_no_show_pattern(sample_contract)
        assert "no_show_suspected" in result
        assert "score" in result

    def test_detect_kickback_routing(self, sample_contract):
        """Test kickback detection."""
        result = sheriff.detect_kickback_routing(sample_contract)
        assert "kickback_suspected" in result

    def test_score_contract_risk(self, sample_contract):
        """Test contract risk scoring."""
        score = sheriff.score_contract_risk(sample_contract)
        assert 0.0 <= score <= 1.0

    def test_run_detection(self, seed):
        """Test full detection run."""
        result = sheriff.run_detection(n_contracts=50, fraud_rate=0.08, seed=seed)
        assert "detection_rate" in result
        assert "receipts" in result


class TestCorporate:
    """Tests for corporate capture detection."""

    def test_generate_donations(self, seed):
        """Test donation generation."""
        donations = corporate.generate_corporate_donations(3, 2, seed)
        assert len(donations) > 0

    def test_generate_policy_outcomes(self, seed):
        """Test policy outcome generation."""
        corps = ["Publix", "FPL/NextEra"]
        policies = corporate.generate_policy_outcomes(5, corps, seed)
        assert len(policies) == 5

    def test_compute_roi(self):
        """Test ROI calculation."""
        roi = corporate.compute_roi_ratio(100000, 10000000)
        assert roi == 100.0

    def test_run_detection(self, seed):
        """Test full detection run."""
        result = corporate.run_detection(
            n_corps=3, n_cycles=2, n_policies=10, fraud_rate=0.12, seed=seed
        )
        assert "detection_rate" in result
        assert "avg_roi" in result


class TestPandemic:
    """Tests for pandemic fraud detection."""

    def test_generate_ppp_applications(self, seed):
        """Test PPP application generation."""
        apps = pandemic.generate_ppp_applications(20, 0.05, seed)
        assert len(apps) == 20

    def test_generate_unemployment_claims(self, seed):
        """Test unemployment claim generation."""
        claims = pandemic.generate_unemployment_claims(20, 0.05, seed)
        assert len(claims) == 20

    def test_detect_ppp_fraud(self, seed):
        """Test PPP fraud detection."""
        apps = pandemic.generate_ppp_applications(10, 0.5, seed)
        for app in apps:
            result = pandemic.detect_ppp_fraud(app)
            assert "fraud_suspected" in result
            assert "score" in result

    def test_aggregate_fraud_patterns(self, seed):
        """Test fraud pattern aggregation."""
        frauds = [
            {"fraud_suspected": True, "fraud_type": "ppp", "indicators": {"network_member": True}},
            {"fraud_suspected": True, "fraud_type": "ppp", "indicators": {"network_member": True}},
            {"fraud_suspected": False, "fraud_type": "unemployment", "indicators": {}},
        ]
        result = pandemic.aggregate_fraud_patterns(frauds)
        assert "systemic" in result
        assert "patterns" in result

    def test_run_detection(self, seed):
        """Test full detection run."""
        result = pandemic.run_detection(
            n_ppp=50, n_unemployment=50, fraud_rate=0.05, seed=seed
        )
        assert "detection_rate" in result
        assert "aggregation" in result
