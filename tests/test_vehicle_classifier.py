import pytest

from backend.processing.vehicle_classifier import (
    FineGrainedVehicleClassifier,
    VehicleConsensus,
)


def test_vehicle_label_parser_extracts_make_model_and_year():
    prediction = FineGrainedVehicleClassifier.parse_label("Honda Accord Sedan 2012", 0.81)
    assert prediction.make == "Honda"
    assert prediction.model == "Accord Sedan"
    assert prediction.year == "2012"


def test_vehicle_consensus_uses_weighted_votes():
    consensus = VehicleConsensus()
    consensus.add(FineGrainedVehicleClassifier.parse_label("Honda Civic 2012", 0.55))
    consensus.add(FineGrainedVehicleClassifier.parse_label("Honda Civic 2012", 0.65))
    consensus.add(FineGrainedVehicleClassifier.parse_label("Toyota Camry 2012", 0.70))
    best = consensus.best()
    assert best is not None
    assert best.label == "Honda Civic 2012"
    assert best.confidence == pytest.approx(0.60)
