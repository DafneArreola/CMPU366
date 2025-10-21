#!/usr/bin/env python3

import tempfile
from pathlib import Path

import pytest
import torch

from asmt5 import get_predicted_label_from_predictions


def test_get_predicted_label_from_predictions():
    """Test that the function returns the index of the maximum value."""
    predictions = torch.tensor([[0.1, 0.7, 0.2]])
    assert get_predicted_label_from_predictions(predictions) == 1

    predictions = torch.tensor([[0.9, 0.05, 0.05]])
    assert get_predicted_label_from_predictions(predictions) == 0

    predictions = torch.tensor([[0.1, 0.1, 0.8]])
    assert get_predicted_label_from_predictions(predictions) == 2
