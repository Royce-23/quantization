import json
from pathlib import Path

import numpy as np

from src.model import build_model
from src.prepare_data import split_and_normalize
from src.train import _learning_rate, _macro_f1
from src.visualize_tv1 import _confusion_matrix


ROOT = Path(__file__).resolve().parents[1]


def test_learning_rate_matches_paper_milestones():
    milestones = [100, 200, 250]
    assert _learning_rate(0, 0.05, milestones, 0.13) == 0.05
    assert np.isclose(_learning_rate(99, 0.05, milestones, 0.13), 0.0065)
    assert np.isclose(_learning_rate(199, 0.05, milestones, 0.13), 0.000845)
    assert np.isclose(_learning_rate(249, 0.05, milestones, 0.13), 0.00010985)


def test_subject_split_and_normalization_do_not_leak():
    rng = np.random.default_rng(7)
    x_train = rng.normal(size=(12, 128, 9)).astype(np.float32)
    y_train = np.tile(np.arange(6), 2)
    subjects_train = np.repeat([1, 2, 3], 4)
    x_test = rng.normal(size=(6, 128, 9)).astype(np.float32)
    y_test = np.arange(6)
    subjects_test = np.repeat(4, 6)
    arrays, _, _ = split_and_normalize(
        (x_train, y_train, subjects_train),
        (x_test, y_test, subjects_test),
        [3],
    )
    assert set(arrays["subject_train"]) == {1, 2}
    assert set(arrays["subject_val"]) == {3}
    assert set(arrays["subject_test"]) == {4}
    np.testing.assert_allclose(arrays["x_train"].mean((0, 1)), 0, atol=1e-6)
    np.testing.assert_allclose(arrays["x_train"].std((0, 1)), 1, atol=1e-5)


def test_model_contract_and_macro_f1():
    config = json.loads((ROOT / "configs/project.json").read_text())
    model = build_model(config)
    assert model.input_shape == (None, 128, 9)
    assert model.output_shape == (None, 6)
    assert model.count_params() == 10726
    labels = np.arange(6)
    assert _macro_f1(labels, labels, 6) == 1.0


def test_visualization_confusion_matrix_counts_predictions():
    y_true = np.array([0, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 0])
    matrix = _confusion_matrix(np, y_true, y_pred, 3)
    np.testing.assert_array_equal(
        matrix,
        np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]]),
    )
