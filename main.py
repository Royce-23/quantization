"""Điều phối module. Chạy --check để xác nhận bộ khung mà không cần thư viện AI."""
import argparse
import importlib
import sys
from src.common import ROOT, load_config

STAGES = {
    "prepare": "prepare_data", "train": "train", "visualize-tv1": "visualize_tv1",
    "visualize-pipeline": "visualize_pipeline",
    "quantize": "quantize", "evaluate": "evaluate",
    "summarize": "summarize_results", "plot": "plot_results",
    "export-c": "export_c",
}


def main():
    parser = argparse.ArgumentParser(description="Quantization HAR — bộ khung nhóm 3 người")
    parser.add_argument("stage", nargs="?", choices=STAGES)
    parser.add_argument("--config", default="configs/project.json")
    parser.add_argument("--check", action="store_true", help="Kiểm tra bộ khung, không chạy AI")
    args = parser.parse_args()
    if not args.check and not args.stage:
        parser.print_help()
        return 0
    try:
        config = load_config(args.config)
        if args.check:
            for module_name in STAGES.values():
                module = importlib.import_module(f"src.{module_name}")
                if not callable(getattr(module, "run", None)):
                    raise ValueError(f"{module_name} thiếu run(config)")
            for folder in config["paths"].values():
                if not (ROOT / folder).is_dir():
                    raise ValueError(f"Thiếu thư mục: {folder}")
            print("OK: cấu hình, thư mục và giao diện module của bộ khung hợp lệ.")
            print("TV1, TV2 và TV3 đã được triển khai; chạy từng stage để kiểm tra artifact.")
            return 0
        module = importlib.import_module(f"src.{STAGES[args.stage]}")
        module.run(config)
        return 0
    except NotImplementedError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (ValueError, OSError, ImportError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
