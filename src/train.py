"""TV1 — huấn luyện và bàn giao baseline FP32 theo cấu hình paper."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
from pathlib import Path

from .common import project_path
from .model import build_model


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _learning_rate(epoch, initial, milestones, factor):
    """Keras dùng epoch từ 0; paper đánh số epoch từ 1."""
    completed = sum((epoch + 1) >= int(milestone) for milestone in milestones)
    return float(initial) * float(factor) ** completed


def _write_history(history, destination):
    keys = list(history)
    with Path(destination).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", *keys])
        for epoch, values in enumerate(zip(*(history[key] for key in keys)), start=1):
            writer.writerow([epoch, *values])


def _macro_f1(y_true, y_pred, class_count):
    scores = []
    for class_id in range(class_count):
        true_positive = ((y_true == class_id) & (y_pred == class_id)).sum()
        false_positive = ((y_true != class_id) & (y_pred == class_id)).sum()
        false_negative = ((y_true == class_id) & (y_pred != class_id)).sum()
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(sum(scores) / len(scores))


def _make_mixup_sequence(tf, np, x, y, batch_size, class_count, alpha, seed):
    class MixupSequence(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(x) / batch_size))

        def __getitem__(self, batch_index):
            positions = slice(batch_index * batch_size, (batch_index + 1) * batch_size)
            first = self.order[positions]
            second = self.partner[positions]
            lam = self.lam[positions].reshape(-1, 1, 1).astype(np.float32)
            mixed_x = lam * x[first] + (1.0 - lam) * x[second]
            y_first = tf.one_hot(y[first], class_count).numpy()
            y_second = tf.one_hot(y[second], class_count).numpy()
            label_lam = lam.reshape(-1, 1)
            mixed_y = label_lam * y_first + (1.0 - label_lam) * y_second
            return mixed_x.astype(np.float32), mixed_y.astype(np.float32)

        def on_epoch_end(self):
            self.order = self.rng.permutation(len(x))
            self.partner = self.rng.permutation(len(x))
            self.lam = self.rng.beta(alpha, alpha, size=len(x))

    return MixupSequence()


def run(config):
    import numpy as np
    import tensorflow as tf

    seed = int(config["seed"])
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass

    processed_dir = project_path(config["paths"]["processed_data"])
    dataset_path = processed_dir / "har.npz"
    if not dataset_path.is_file():
        raise FileNotFoundError("Thiếu data/processed/har.npz; chạy 'python main.py prepare'")
    with np.load(dataset_path) as dataset:
        x_train = dataset["x_train"]
        y_train = dataset["y_train"]
        x_val = dataset["x_val"]
        y_val = dataset["y_val"]
        x_test = dataset["x_test"]
        y_test = dataset["y_test"]

    training = config["training"]
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])
    class_count = len(config["dataset"]["labels"])
    mixup = _make_mixup_sequence(
        tf,
        np,
        x_train,
        y_train,
        batch_size,
        class_count,
        float(training["mixup_alpha"]),
        seed,
    )

    model = build_model(config)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=float(training["learning_rate"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    model_dir = project_path(config["paths"]["models"])
    results_dir = project_path(config["paths"]["results"])
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "baseline.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor=training["checkpoint_monitor"],
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: _learning_rate(
                epoch,
                training["learning_rate"],
                training["lr_milestones"],
                training["lr_decay_factor"],
            ),
            verbose=1,
        ),
    ]
    history = model.fit(
        mixup,
        validation_data=(x_val, tf.one_hot(y_val, class_count)),
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    history_path = results_dir / "tv1_training_history.csv"
    _write_history(history.history, history_path)

    best_model = tf.keras.models.load_model(checkpoint_path)
    val_logits = best_model.predict(x_val, batch_size=batch_size, verbose=0)
    test_logits = best_model.predict(x_test, batch_size=batch_size, verbose=0)
    val_predictions = val_logits.argmax(axis=1)
    test_predictions = test_logits.argmax(axis=1)
    best_epoch = int(np.argmax(history.history[training["checkpoint_monitor"]]) + 1)
    metrics = {
        "validation_accuracy": float(np.mean(val_predictions == y_val)),
        "validation_macro_f1": _macro_f1(y_val, val_predictions, class_count),
        "test_accuracy": float(np.mean(test_predictions == y_test)),
        "test_macro_f1": _macro_f1(y_test, test_predictions, class_count),
        "best_epoch": best_epoch,
        "epochs_ran": epochs,
        "parameter_count": int(best_model.count_params()),
        "keras_model_size_bytes": checkpoint_path.stat().st_size,
    }

    checkpoint_id = _sha256(checkpoint_path)
    metadata = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_file": "models/baseline.keras",
        "seed": seed,
        "architecture": config["model"],
        "input_shape": config["dataset"]["input_shape"],
        "channel_order": config["dataset"]["channel_order"],
        "labels": config["dataset"]["labels"],
        "training": training,
        "selection_split": "validation",
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
        },
        "paper_alignment": {
            "source": "https://doi.org/10.3390/s21092984",
            "matched": [
                "ResNetv1-6 Figure 4 adapted to Conv1D/MaxPool1D",
                "z-score using model-training data",
                "SGD, batch 64, 300 epochs, lr 0.05, momentum 0.9",
                "weight decay 5e-4 and lr factor 0.13 at epochs 100/200/250",
                "Mixup training",
            ],
            "project_changes": [
                "32 filters selected from paper sweep",
                "subject-disjoint validation split for checkpoint selection",
                "Mixup alpha 0.2 because the paper does not report alpha",
                "project studies TFLite PTQ INT8 rather than reproducing MicroAI QAT",
            ],
        },
    }
    (model_dir / "baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    sanity_path = processed_dir / "sanity_samples.npz"
    with np.load(sanity_path) as sanity:
        sanity_x = sanity["x"]
        sanity_y = sanity["y"]
        sanity_indices = sanity["validation_indices"]
    sanity_logits = best_model.predict(sanity_x, verbose=0)
    np.savez_compressed(
        sanity_path,
        x=sanity_x,
        y=sanity_y,
        validation_indices=sanity_indices,
        baseline_logits=sanity_logits,
        baseline_predictions=sanity_logits.argmax(axis=1),
        checkpoint_id=np.asarray(checkpoint_id),
    )
    print(json.dumps({"checkpoint_id": checkpoint_id, **metrics}, indent=2))
    return checkpoint_path
