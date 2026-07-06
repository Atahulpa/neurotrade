"""Tests pour models/ — déterminisme, shapes, interfaces.

DoD étape 4 :
- Entraînement déterministe (même seed → mêmes poids)
- Courbes de perte : val_loss diminue au moins quelques epochs
- predict_proba retourne des probabilités valides (somme=1, ∈[0,1])
"""

from __future__ import annotations

import numpy as np
import torch

from neurotrade.config.schema import ModelConfig
from neurotrade.models.mlp import MLP
from neurotrade.models.trainer import Trainer


def _fix_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _make_data(
    n_train: int = 300,
    n_val: int = 50,
    n_features: int = 6,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_train, n_features)).astype(np.float32)
    y_train = rng.choice([-1, 0, 1], size=n_train).astype(np.float64)
    X_val = rng.normal(size=(n_val, n_features)).astype(np.float32)
    y_val = rng.choice([-1, 0, 1], size=n_val).astype(np.float64)
    return X_train, y_train, X_val, y_val


class TestMLPInterface:
    def test_predict_proba_shape(self) -> None:
        _fix_seed()
        cfg = ModelConfig(arch="mlp", hidden_dims=[16, 8], max_epochs=2, patience=5)
        model = MLP(input_dim=6, config=cfg)
        X_train, y_train, X_val, y_val = _make_data()
        trainer = Trainer(model, cfg)
        trainer.train(X_train, y_train, X_val, y_val)

        X_test = np.random.default_rng(99).normal(size=(20, 6)).astype(np.float32)
        proba = model.predict_proba(X_test)
        assert proba.shape == (20, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert (proba >= 0.0).all()
        assert (proba <= 1.0).all()

    def test_predict_retourne_labels_valides(self) -> None:
        _fix_seed()
        cfg = ModelConfig(arch="mlp", hidden_dims=[8], max_epochs=2, patience=5)
        model = MLP(input_dim=6, config=cfg)
        X_train, y_train, X_val, y_val = _make_data()
        Trainer(model, cfg).train(X_train, y_train, X_val, y_val)

        X_test = np.random.default_rng(1).normal(size=(30, 6)).astype(np.float32)
        preds = model.predict(X_test)
        assert set(preds).issubset({-1, 0, 1})


class TestDeterminisme:
    """Même seed → mêmes poids après entraînement (DoD étape 4)."""

    def test_mlp_deterministe(self) -> None:
        cfg = ModelConfig(arch="mlp", hidden_dims=[16, 8], max_epochs=5, patience=10)
        X_train, y_train, X_val, y_val = _make_data()

        # Run 1
        _fix_seed(42)
        model1 = MLP(input_dim=6, config=cfg)
        Trainer(model1, cfg).train(X_train, y_train, X_val, y_val)
        proba1 = model1.predict_proba(X_val.astype(np.float32))

        # Run 2 — même seed
        _fix_seed(42)
        model2 = MLP(input_dim=6, config=cfg)
        Trainer(model2, cfg).train(X_train, y_train, X_val, y_val)
        proba2 = model2.predict_proba(X_val.astype(np.float32))

        np.testing.assert_allclose(proba1, proba2, atol=1e-5,
                                   err_msg="Entraînement non déterministe malgré même seed.")


class TestTrainerEarlyStopping:
    def test_val_loss_retournee(self) -> None:
        _fix_seed()
        cfg = ModelConfig(arch="mlp", hidden_dims=[16], max_epochs=10, patience=3)
        model = MLP(input_dim=6, config=cfg)
        X_train, y_train, X_val, y_val = _make_data()
        losses = Trainer(model, cfg).train(X_train, y_train, X_val, y_val)
        assert len(losses) > 0
        assert all(isinstance(v, float) for v in losses)

    def test_early_stopping_respecte_patience(self) -> None:
        """Avec patience=2 et max_epochs=100, on doit s'arrêter bien avant 100 epochs.

        On utilise un très petit set d'entraînement (10 samples) pour que le modèle
        sur-apprenne rapidement, faisant remonter la val_loss et déclenchant l'early stopping.
        """
        _fix_seed()
        cfg = ModelConfig(arch="mlp", hidden_dims=[32, 16], max_epochs=100, patience=2, lr=0.05)
        model = MLP(input_dim=6, config=cfg)
        rng = np.random.default_rng(42)
        # Tiny train set → overfitting, large val set on different distribution
        X_train = rng.normal(0.0, 1.0, size=(10, 6)).astype(np.float32)
        y_train = rng.choice([-1, 0, 1], size=10).astype(np.float64)
        X_val = rng.normal(5.0, 1.0, size=(200, 6)).astype(np.float32)  # distribution shift
        y_val = rng.choice([-1, 0, 1], size=200).astype(np.float64)
        losses = Trainer(model, cfg).train(X_train, y_train, X_val, y_val)
        assert len(losses) < 100, "Early stopping n'a pas arrêté l'entraînement."


class TestCNN1D:
    def test_predict_proba_shape_sequence(self) -> None:
        from neurotrade.models.cnn1d import CNN1D

        _fix_seed()
        cfg = ModelConfig(arch="cnn1d", max_epochs=2, patience=5, window_size=10)
        model = CNN1D(n_features=6, config=cfg)

        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 10, 6)).astype(np.float32)  # (batch, seq_len, features)
        y = rng.choice([-1, 0, 1], size=50).astype(np.float64)
        X_val = rng.normal(size=(10, 10, 6)).astype(np.float32)
        y_val = rng.choice([-1, 0, 1], size=10).astype(np.float64)

        Trainer(model, cfg).train(X, y, X_val, y_val)
        proba = model.predict_proba(X_val)
        assert proba.shape == (10, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestLSTM:
    def test_predict_proba_shape_sequence(self) -> None:
        from neurotrade.models.lstm import LSTMModel

        _fix_seed()
        cfg = ModelConfig(arch="lstm", max_epochs=2, patience=5, window_size=10)
        model = LSTMModel(n_features=6, config=cfg)

        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 10, 6)).astype(np.float32)
        y = rng.choice([-1, 0, 1], size=40).astype(np.float64)
        X_val = rng.normal(size=(10, 10, 6)).astype(np.float32)
        y_val = rng.choice([-1, 0, 1], size=10).astype(np.float64)

        Trainer(model, cfg).train(X, y, X_val, y_val)
        proba = model.predict_proba(X_val)
        assert proba.shape == (10, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
