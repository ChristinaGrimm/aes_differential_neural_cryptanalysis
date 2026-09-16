import csv
import os
import time

import numpy as np
import tensorflow as tf

import aes_cipher as aes
from attack_utils import (
    column_flat_indices,
    configure_tensorflow,
    diagonal_indices,
    encrypt_with_one_key,
    enumerate_restricted_diagonal,
    full_key_candidates,
    generate_attack_data,
    invert_aes128_round_key,
    key_hex,
    load_distinguisher,
    make_output_paths,
    response_metrics,
    score_candidates_for_counts,
    write_json,
)


CONFIG = {
    # One m=32 Method 2 model for each peeled byte column
    "model_files": [
        "saved_models/part-state/0-3/aes_r3_gs32_dp0_80_sel0x1x2x3_d5_invmc_adv_best_0.6967.h5",
        "saved_models/part-state/4-7/aes_r3_gs32_dp0_80_sel4x5x6x7_d5_invmc_adv_best_0.6974.h5",
        "saved_models/part-state/8-11/aes_r3_gs32_dp0_80_sel8x9x10x11_d5_invmc_adv_best_0.6965.h5",
        "saved_models/part-state/12-15/aes_r3_gs32_dp0_80_sel12x13x14x15_d5_invmc_adv_best_0.6972.h5",
    ],
    "group_size": 32,
    "depth": 5,
    # Four-round restricted-space experiment
    "attack_rounds": 4,
    "diff": [0x80] + [0x00] * 15,
    "n_instances": 30,
    "data_seed": 20260910,
    # Seventeen unknown bits and fifteen known bits per 32-bit diagonal
    "unknown_bit_positions": [
        0,
        1,
        2,
        3,
        4,
        8,
        9,
        10,
        11,
        16,
        17,
        18,
        19,
        24,
        25,
        26,
        27,
    ],
    # Memory and prediction parameters
    "candidate_batch_size": 256,
    "prediction_batch_size": 1024,
    # Fresh-pair verification and output parameters
    "verification_pairs": 4,
    "verification_seed": 314159,
    "output_dir": "restricted_attack_results",
}


def validate_config(config):
    if len(config["model_files"]) != 4:
        raise ValueError("model_files must contain four column models")
    if config["group_size"] < 1 or config["n_instances"] < 1:
        raise ValueError("group_size and n_instances must be positive")
    if config["verification_pairs"] < 1:
        raise ValueError("verification_pairs must be positive")
    if len(config["diff"]) != 16:
        raise ValueError("diff must contain 16 bytes")


def write_diagonal_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    # The default parameters reproduce the restricted 2^17-candidate experiment.
    validate_config(CONFIG)
    print(f"GPU count: {configure_tensorflow()}")
    run_config = dict(CONFIG)
    run_config["max_instances"] = CONFIG["n_instances"]
    data = generate_attack_data(run_config)
    true_last_round_key = data["last_round_key"]
    recovered_last_round_key = np.zeros(16, dtype=np.uint8)
    rows = []
    profiles = []
    candidate_values_by_diagonal = []
    started = time.time()
    for diagonal_index, model_file in enumerate(CONFIG["model_files"]):
        key_indices = diagonal_indices(diagonal_index)
        true_diagonal = true_last_round_key[key_indices]
        diagonal_values, true_index = enumerate_restricted_diagonal(
            true_diagonal,
            CONFIG["unknown_bit_positions"],
        )
        full_candidates = full_key_candidates(diagonal_values, diagonal_index)
        selected_bytes = column_flat_indices(diagonal_index)
        model = load_distinguisher(
            model_file,
            CONFIG["group_size"],
            selected_bytes,
            CONFIG["depth"],
        )
        _, diagonal_profiles, seconds = score_candidates_for_counts(
            model=model,
            ciphertext0=data["ciphertext0"],
            ciphertext1=data["ciphertext1"],
            candidates=full_candidates,
            diagonal_index=diagonal_index,
            group_size=CONFIG["group_size"],
            instance_counts=[CONFIG["n_instances"]],
            candidate_batch_size=CONFIG["candidate_batch_size"],
            prediction_batch_size=CONFIG["prediction_batch_size"],
            progress_label=f"diagonal={diagonal_index}",
        )
        profile = diagonal_profiles[0]
        metrics = response_metrics(profile, true_index)
        predicted_diagonal = diagonal_values[metrics["predicted_index"]]
        recovered_last_round_key[key_indices] = predicted_diagonal
        row = {
            "diagonal_index": diagonal_index,
            "model_file": model_file,
            "key_byte_indices": str(key_indices),
            "true_diagonal": key_hex(true_diagonal),
            "predicted_diagonal": key_hex(predicted_diagonal),
            "true_candidate_index": true_index,
            "predicted_candidate_index": metrics["predicted_index"],
            "correct_selected": metrics["correct_selected"],
            "true_response": metrics["true_response"],
            "wrong_mean": metrics["wrong_mean"],
            "wrong_std": metrics["wrong_std"],
            "wrong_max": metrics["wrong_max"],
            "response_margin": metrics["response_margin"],
            "separation": metrics["separation"],
            "seconds": seconds,
        }
        rows.append(row)
        profiles.append(profile)
        candidate_values_by_diagonal.append(diagonal_values)
        print(
            f"diagonal={diagonal_index} true={row['true_diagonal']} "
            f"predicted={row['predicted_diagonal']} "
            f"correct={row['correct_selected']}"
        )
        del model, full_candidates
        tf.keras.backend.clear_session()
    recovered_master_key = invert_aes128_round_key(
        recovered_last_round_key,
        CONFIG["attack_rounds"],
    )
    recovered_schedule = aes.expand_key(
        recovered_master_key.reshape(1, 16),
        CONFIG["attack_rounds"],
    )
    schedule_consistent = np.array_equal(
        recovered_schedule[-1, 0],
        recovered_last_round_key,
    )
    master_key_match = np.array_equal(recovered_master_key, data["master_key"])
    rng = np.random.RandomState(CONFIG["verification_seed"])
    verification_plaintexts = rng.randint(
        0,
        256,
        size=(CONFIG["verification_pairs"], 16),
        dtype=np.uint8,
    )
    true_ciphertexts = encrypt_with_one_key(
        verification_plaintexts,
        data["master_key"],
        CONFIG["attack_rounds"],
    )
    recovered_ciphertexts = encrypt_with_one_key(
        verification_plaintexts,
        recovered_master_key,
        CONFIG["attack_rounds"],
    )
    verification_passed = np.array_equal(true_ciphertexts, recovered_ciphertexts)
    all_diagonals_correct = all(row["correct_selected"] for row in rows)
    path = make_output_paths(CONFIG["output_dir"], "restricted_end_to_end")
    csv_path = path("csv")
    npz_path = path("npz")
    json_path = path("json")
    write_diagonal_csv(rows, csv_path)
    np.savez_compressed(
        npz_path,
        profiles=np.stack(profiles),
        diagonal_candidates=np.stack(candidate_values_by_diagonal),
        true_master_key=data["master_key"],
        recovered_master_key=recovered_master_key,
        true_last_round_key=true_last_round_key,
        recovered_last_round_key=recovered_last_round_key,
        verification_plaintexts=verification_plaintexts,
        verification_ciphertexts=true_ciphertexts,
    )
    summary = {
        "config": CONFIG,
        "candidate_count_per_diagonal": 1
        << len(CONFIG["unknown_bit_positions"]),
        "known_bits_per_diagonal": 32 - len(CONFIG["unknown_bit_positions"]),
        "unknown_bits_per_diagonal": len(CONFIG["unknown_bit_positions"]),
        "true_master_key": key_hex(data["master_key"]),
        "recovered_master_key": key_hex(recovered_master_key),
        "true_last_round_key": key_hex(true_last_round_key),
        "recovered_last_round_key": key_hex(recovered_last_round_key),
        "all_diagonals_correct": all_diagonals_correct,
        "master_key_match": master_key_match,
        "inverse_schedule_consistent": schedule_consistent,
        "fresh_pair_verification_passed": verification_passed,
        "diagonal_metrics": rows,
        "seconds": time.time() - started,
        "csv_file": os.path.basename(csv_path),
        "npz_file": os.path.basename(npz_path),
    }
    write_json(json_path, summary)
    print(f"True master key:      {summary['true_master_key']}")
    print(f"Recovered master key: {summary['recovered_master_key']}")
    print(f"All diagonals correct: {all_diagonals_correct}")
    print(f"Master key match:      {master_key_match}")
    print(f"Fresh-pair check:      {verification_passed}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
