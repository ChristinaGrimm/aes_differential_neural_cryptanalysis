import json
import os
import time

import numpy as np

import AES_data_gen as data_gen
from attack_utils import configure_tensorflow, load_distinguisher, resolve_path


HERE = os.path.dirname(os.path.abspath(__file__))


CONFIG = {
    # Model parameters
    "model_file": "saved_models/full-state/aes_r3_gs32_d0x80_p0_sel0x1x2x3x4x5x6x7x8x9x10x11x12x13x14x15_d5_invmc_adv_best_0.9323.h5",
    "group_size": 32,
    "selected_bytes": list(range(16)),
    "depth": 5,
    # Method 2 evaluation data
    "num_rounds": 3,
    "diff": [0x80] + [0x00] * 15,
    "n_eval_samples": 100_000,
    "generation_chunk_samples": 2_000,
    # Prediction and output parameters
    "prediction_batch_size": 2_000,
    "threshold": 0.5,
    "output_file": "evaluation_full_state.json",
}


def validate_config(config):
    if config["group_size"] < 1:
        raise ValueError("group_size must be positive")
    if config["n_eval_samples"] < 2 or config["n_eval_samples"] % 2:
        raise ValueError("n_eval_samples must be a positive even integer")
    if config["generation_chunk_samples"] < 2:
        raise ValueError("generation_chunk_samples must be at least 2")
    if len(config["diff"]) != 16:
        raise ValueError("diff must contain 16 bytes")


def confusion_counts(labels, predictions, threshold):
    true_values = np.asarray(labels, dtype=np.uint8)
    predicted_values = np.asarray(predictions) >= threshold
    return {
        "tp": int(np.sum((true_values == 1) & predicted_values)),
        "fn": int(np.sum((true_values == 1) & ~predicted_values)),
        "tn": int(np.sum((true_values == 0) & ~predicted_values)),
        "fp": int(np.sum((true_values == 0) & predicted_values)),
    }


def metrics_from_counts(counts):
    positive = counts["tp"] + counts["fn"]
    negative = counts["tn"] + counts["fp"]
    total = positive + negative
    return {
        **counts,
        "tpr": counts["tp"] / positive if positive else 0.0,
        "tnr": counts["tn"] / negative if negative else 0.0,
        "fnr": counts["fn"] / positive if positive else 0.0,
        "fpr": counts["fp"] / negative if negative else 0.0,
        "accuracy": (counts["tp"] + counts["tn"]) / total if total else 0.0,
        "samples": total,
    }


def evaluate_model(config):
    validate_config(config)
    model = load_distinguisher(
        config["model_file"],
        config["group_size"],
        config["selected_bytes"],
        config["depth"],
    )
    remaining = config["n_eval_samples"]
    aggregate = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    positive_scores = []
    negative_scores = []
    started = time.time()
    while remaining:
        chunk_samples = min(config["generation_chunk_samples"], remaining)
        if chunk_samples % 2:
            chunk_samples -= 1
        if chunk_samples < 2:
            chunk_samples = 2
        pair_count = chunk_samples * config["group_size"]
        inputs, labels = data_gen.make_dataset_with_group_size(
            pair_count,
            config["num_rounds"],
            diff=config["diff"],
            group_size=config["group_size"],
            selected_bytes=config["selected_bytes"],
        )
        predictions = model.predict(
            inputs.astype(np.float32),
            batch_size=config["prediction_batch_size"],
            verbose=0,
        ).reshape(-1)
        counts = confusion_counts(labels, predictions, config["threshold"])
        for key in aggregate:
            aggregate[key] += counts[key]
        positive_scores.append(predictions[labels == 1])
        negative_scores.append(predictions[labels == 0])
        remaining -= chunk_samples
        completed = config["n_eval_samples"] - remaining
        print(f"Evaluated {completed:,}/{config['n_eval_samples']:,} samples")
    positive = np.concatenate(positive_scores)
    negative = np.concatenate(negative_scores)
    result = metrics_from_counts(aggregate)
    result.update(
        {
            "positive_mean": float(np.mean(positive)),
            "positive_std": float(np.std(positive)),
            "negative_mean": float(np.mean(negative)),
            "negative_std": float(np.std(negative)),
            "seconds": time.time() - started,
            "model_file": config["model_file"],
            "group_size": config["group_size"],
            "selected_bytes": config["selected_bytes"],
        }
    )
    return result


def save_result(result, output_file):
    path = resolve_path(output_file)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return path


def main():
    # Edit CONFIG before evaluating a different full-state model.
    gpu_count = configure_tensorflow()
    print(f"GPU count: {gpu_count}")
    result = evaluate_model(CONFIG)
    output_path = save_result(result, CONFIG["output_file"])
    print(f"ACC: {result['accuracy']:.6f}")
    print(f"TPR: {result['tpr']:.6f}")
    print(f"TNR: {result['tnr']:.6f}")
    print(f"FPR: {result['fpr']:.6f}")
    print(f"FNR: {result['fnr']:.6f}")
    print(f"Result: {output_path}")


if __name__ == "__main__":
    main()
