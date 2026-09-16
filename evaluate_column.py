import csv
import glob
import os
import re
import time

import tensorflow as tf

from attack_utils import configure_tensorflow, resolve_path
from evaluate_full_state import evaluate_model


HERE = os.path.dirname(os.path.abspath(__file__))


CONFIG = {
    # Model discovery and fixed Method 2 parameters
    "models_dir": "saved_models/part-state",
    "model_pattern": "*/*.h5",
    "diff": [0x80] + [0x00] * 15,
    "num_rounds": 3,
    # Evaluation parameters
    "n_eval_samples": 10_000,
    "generation_chunk_samples": 2_000,
    "prediction_batch_size": 2_000,
    "threshold": 0.5,
    # Output parameters
    "output_file": "evaluation_columns.csv",
}


def parse_model_filename(model_file):
    name = os.path.basename(model_file)
    match = re.search(
        r"aes_r(\d+)_gs(\d+).*_sel([0-9x]+)_d(\d+).*\.h5$",
        name,
    )
    if not match:
        raise ValueError(f"cannot parse model filename: {name}")
    selected_bytes = [int(value) for value in match.group(3).split("x")]
    return {
        "num_rounds": int(match.group(1)),
        "group_size": int(match.group(2)),
        "selected_bytes": selected_bytes,
        "depth": int(match.group(4)),
    }


def discover_models(config):
    root = resolve_path(config["models_dir"])
    files = sorted(glob.glob(os.path.join(root, config["model_pattern"])))
    if not files:
        raise FileNotFoundError(f"no model files found under {root}")
    return files


def result_row(model_file, params, metrics):
    return {
        "byte_column": os.path.basename(os.path.dirname(model_file)),
        "model_file": os.path.relpath(model_file, HERE),
        "group_size": params["group_size"],
        "selected_bytes": ",".join(map(str, params["selected_bytes"])),
        "depth": params["depth"],
        "samples": metrics["samples"],
        "tp": metrics["tp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "tpr": metrics["tpr"],
        "tnr": metrics["tnr"],
        "fpr": metrics["fpr"],
        "fnr": metrics["fnr"],
        "accuracy": metrics["accuracy"],
        "seconds": metrics["seconds"],
    }


def write_csv(rows, output_file):
    path = resolve_path(output_file)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    # Edit CONFIG to change the evaluation size or model directory.
    gpu_count = configure_tensorflow()
    print(f"GPU count: {gpu_count}")
    models = discover_models(CONFIG)
    rows = []
    started = time.time()
    for index, model_file in enumerate(models, 1):
        params = parse_model_filename(model_file)
        evaluation_config = {
            "model_file": model_file,
            "group_size": params["group_size"],
            "selected_bytes": params["selected_bytes"],
            "depth": params["depth"],
            "num_rounds": CONFIG["num_rounds"],
            "diff": CONFIG["diff"],
            "n_eval_samples": CONFIG["n_eval_samples"],
            "generation_chunk_samples": CONFIG["generation_chunk_samples"],
            "prediction_batch_size": CONFIG["prediction_batch_size"],
            "threshold": CONFIG["threshold"],
        }
        print(
            f"[{index}/{len(models)}] m={params['group_size']} "
            f"bytes={params['selected_bytes']} file={os.path.basename(model_file)}"
        )
        metrics = evaluate_model(evaluation_config)
        row = result_row(model_file, params, metrics)
        rows.append(row)
        print(
            f"ACC={row['accuracy']:.6f} "
            f"TPR={row['tpr']:.6f} TNR={row['tnr']:.6f}"
        )
        tf.keras.backend.clear_session()
    rows.sort(key=lambda row: (row["byte_column"], row["group_size"]))
    output_path = write_csv(rows, CONFIG["output_file"])
    print(f"Evaluated {len(rows)} models in {time.time() - started:.1f} seconds")
    print(f"Result: {output_path}")


if __name__ == "__main__":
    main()
