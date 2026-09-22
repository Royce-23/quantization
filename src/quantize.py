"""TV2 — xuất TFLite FP32 và lượng tử hóa sau huấn luyện full-integer INT8."""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
from pathlib import Path

from .calibration import select_calibration
from .common import project_path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_metadata(detail):
    import numpy as np

    quant = detail.get("quantization_parameters", {})
    scales = np.asarray(quant.get("scales", []))
    zero_points = np.asarray(quant.get("zero_points", []))
    return {
        "name": detail["name"],
        "shape": [int(value) for value in detail["shape"]],
        "shape_signature": [int(value) for value in detail.get("shape_signature", [])],
        "dtype": np.dtype(detail["dtype"]).name,
        "scales": [float(value) for value in scales],
        "zero_points": [int(value) for value in zero_points],
        "quantized_dimension": int(quant.get("quantized_dimension", 0)),
    }


def _inspect_tflite(tf, np, model_path, expect_int8):
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
        experimental_preserve_all_tensors=True,
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    tensor_details = interpreter.get_tensor_details()
    float_tensors = [
        detail["name"]
        for detail in tensor_details
        if np.dtype(detail["dtype"]) == np.dtype(np.float32)
    ]
    ops = [item["op_name"] for item in interpreter._get_ops_details()]

    input_meta = _tensor_metadata(input_detail)
    output_meta = _tensor_metadata(output_detail)
    full_integer = (
        input_meta["dtype"] == "int8"
        and output_meta["dtype"] == "int8"
        and not float_tensors
    )
    if expect_int8 and not full_integer:
        raise ValueError(
            "Model INT8 còn tensor float hoặc input/output chưa phải int8: "
            f"input={input_meta['dtype']}, output={output_meta['dtype']}, "
            f"float_tensors={float_tensors[:5]}"
        )
    return {
        "input": input_meta,
        "output": output_meta,
        "operators": ops,
        "float_tensor_count": len(float_tensors),
        "full_integer": full_integer,
    }


def _representative_dataset(samples):
    import numpy as np

    def generate():
        for sample in samples:
            yield [np.asarray(sample[np.newaxis, ...], dtype=np.float32)]

    return generate


def run(config):
    import numpy as np
    import tensorflow as tf

    model_dir = project_path(config["paths"]["models"])
    processed_dir = project_path(config["paths"]["processed_data"])
    checkpoint_path = model_dir / "baseline.keras"
    checkpoint_metadata_path = model_dir / "baseline_metadata.json"
    dataset_path = processed_dir / "har.npz"
    for required in (checkpoint_path, checkpoint_metadata_path, dataset_path):
        if not required.is_file():
            raise FileNotFoundError(f"Thiếu {required.relative_to(project_path('.'))}")

    checkpoint_metadata = json.loads(checkpoint_metadata_path.read_text(encoding="utf-8"))
    checkpoint_id = _sha256(checkpoint_path)
    if checkpoint_id != checkpoint_metadata.get("checkpoint_id"):
        raise ValueError("SHA256 baseline.keras không khớp baseline_metadata.json")

    with np.load(dataset_path) as dataset:
        x_train = dataset["x_train"].astype(np.float32, copy=False)
    if tuple(x_train.shape[1:]) != tuple(config["dataset"]["input_shape"]):
        raise ValueError(f"Shape x_train không đúng cấu hình: {x_train.shape}")

    model = tf.keras.models.load_model(checkpoint_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    # TF 2.16 + Keras 3 có thể lỗi LLVM khi from_keras_model() trực tiếp.
    # Export SavedModel một lần rồi dùng from_saved_model() cho mọi biến thể.
    with tempfile.TemporaryDirectory(prefix="har_saved_model_") as temporary:
        saved_model_dir = Path(temporary) / "saved_model"
        model.export(saved_model_dir, verbose=False)

        fp32_path = model_dir / "baseline_fp32.tflite"
        fp32_bytes = tf.lite.TFLiteConverter.from_saved_model(
            str(saved_model_dir)
        ).convert()
        fp32_path.write_bytes(fp32_bytes)
        fp32_info = _inspect_tflite(tf, np, fp32_path, expect_int8=False)

        artifacts = [
            {
                "model_kind": "fp32",
                "model_file": str(fp32_path.relative_to(project_path('.'))),
                "model_id": _sha256(fp32_path),
                "checkpoint_id": checkpoint_id,
                "calibration_size": None,
                "calibration_seed": None,
                "calibration_indices": [],
                "size_bytes": fp32_path.stat().st_size,
                **fp32_info,
            }
        ]

        quant = config["quantization"]
        sizes = [int(value) for value in quant["calibration_sizes"]]
        seeds = [int(value) for value in quant["calibration_seeds"]]
        if len(set(sizes)) != len(sizes) or len(set(seeds)) != len(seeds):
            raise ValueError("calibration_sizes và calibration_seeds không được trùng")

        for size in sizes:
            for seed in seeds:
                samples, indices = select_calibration(x_train, size, seed)
                converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = _representative_dataset(samples)
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                model_bytes = converter.convert()

                output_path = model_dir / f"int8_n{size}_s{seed}.tflite"
                output_path.write_bytes(model_bytes)
                inspected = _inspect_tflite(tf, np, output_path, expect_int8=True)
                artifacts.append(
                    {
                        "model_kind": "int8",
                        "model_file": str(output_path.relative_to(project_path('.'))),
                        "model_id": _sha256(output_path),
                        "checkpoint_id": checkpoint_id,
                        "calibration_size": size,
                        "calibration_seed": seed,
                        "calibration_indices": [int(value) for value in indices],
                        "size_bytes": output_path.stat().st_size,
                        **inspected,
                    }
                )
                print(
                    f"INT8 n={size}, seed={seed}: {output_path.stat().st_size} byte, "
                    f"input scale={inspected['input']['scales'][0]:.8g}, "
                    f"full_integer={inspected['full_integer']}"
                )

    metadata = {
        "schema_version": 1,
        "checkpoint_id": checkpoint_id,
        "source_checkpoint": "models/baseline.keras",
        "method": "TFLite post-training full-integer quantization",
        "representative_source": "x_train in data/processed/har.npz (already z-score normalized)",
        "selection_policy": (
            "compare calibration sizes by mean validation accuracy across seeds; "
            "evaluate only the selected size on test"
        ),
        "labels": config["dataset"]["labels"],
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
        },
        "models": artifacts,
    }
    metadata_path = model_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Đã xuất {len(artifacts)} model và metadata: {metadata_path}")
    return metadata_path
