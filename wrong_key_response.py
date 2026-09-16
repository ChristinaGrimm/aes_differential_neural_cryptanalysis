import os

import numpy as np
import tensorflow as tf

from attack_utils import (
    candidates_from_deltas,
    column_flat_indices,
    configure_tensorflow,
    generate_attack_data,
    generate_deltas,
    load_distinguisher,
    make_output_paths,
    response_metrics,
    score_candidates_for_counts,
    write_json,
)


try:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


HERE = os.path.dirname(os.path.abspath(__file__))


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
    # Four-round response-profile parameters
    "attack_rounds": 4,
    "diff": [0x80] + [0x00] * 15,
    "n_instances": 50,
    "wrong_key_count": 2**17,
    "delta_mode": "contiguous",
    "delta_seed": 12345,
    "data_seed": 20260910,
    # Memory and prediction parameters
    "candidate_batch_size": 256,
    "prediction_batch_size": 1024,
    # Output parameters
    "output_dir": "wrong_key_results",
    "save_png": True,
    "save_eps": True,
    "figure_width_in": 3.2,
    "figure_height_in": 2.25,
    "png_dpi": 600,
}


def validate_config(config):
    if len(config["model_files"]) != 4:
        raise ValueError("model_files must contain four column models")
    if config["group_size"] < 1 or config["n_instances"] < 1:
        raise ValueError("group_size and n_instances must be positive")
    if len(config["diff"]) != 16:
        raise ValueError("diff must contain 16 bytes")


def plot_profile(profile, path, config, title):
    if plt is None:
        return
    x_values = np.arange(len(profile))
    figure, axis = plt.subplots(
        figsize=(config["figure_width_in"], config["figure_height_in"])
    )
    axis.plot(x_values, profile, color="steelblue", linewidth=0.9, label="Mean Response")
    axis.scatter(
        [0],
        [profile[0]],
        color="crimson",
        s=24,
        zorder=5,
        label=f"True key: {profile[0]:.4f}",
    )
    axis.set_xlim(0, len(profile) - 1)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(
        "Wrong-Key Sample Index"
        if config["delta_mode"] == "random"
        else "Difference to Correct Key",
        fontsize=10,
    )
    axis.set_ylabel("Mean Response", fontsize=10)
    axis.set_title(title, fontsize=9)
    axis.tick_params(axis="both", labelsize=8.5, width=0.7)
    axis.grid(color="0.85", linewidth=0.45)
    axis.legend(fontsize=8, loc="upper right")
    figure.tight_layout(pad=0.35)
    if path.lower().endswith(".png"):
        figure.savefig(path, dpi=config["png_dpi"], bbox_inches="tight", pad_inches=0.02)
    else:
        figure.savefig(path, format="eps", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def run_column(config, diagonal_index, model_file, deltas):
    run_config = dict(config)
    run_config["max_instances"] = config["n_instances"]
    data = generate_attack_data(run_config, seed_offset=diagonal_index)
    candidates = candidates_from_deltas(
        data["last_round_key"],
        diagonal_index,
        deltas,
    )
    selected_bytes = column_flat_indices(diagonal_index)
    model = load_distinguisher(
        model_file,
        config["group_size"],
        selected_bytes,
        config["depth"],
    )
    _, profiles, seconds = score_candidates_for_counts(
        model=model,
        ciphertext0=data["ciphertext0"],
        ciphertext1=data["ciphertext1"],
        candidates=candidates,
        diagonal_index=diagonal_index,
        group_size=config["group_size"],
        instance_counts=[config["n_instances"]],
        candidate_batch_size=config["candidate_batch_size"],
        prediction_batch_size=config["prediction_batch_size"],
        progress_label=f"diagonal={diagonal_index}",
    )
    profile = profiles[0]
    metrics = response_metrics(profile, true_index=0)
    metrics.update(
        {
            "diagonal_index": diagonal_index,
            "selected_bytes": selected_bytes,
            "model_file": model_file,
            "seconds": seconds,
        }
    )
    return profile, metrics


def main():
    # Use delta_mode="contiguous" and delta_mode="random" in separate runs.
    validate_config(CONFIG)
    print(f"GPU count: {configure_tensorflow()}")
    deltas = generate_deltas(
        CONFIG["wrong_key_count"],
        CONFIG["delta_mode"],
        CONFIG["delta_seed"],
    )
    path = make_output_paths(
        CONFIG["output_dir"],
        f"wrong_key_{CONFIG['delta_mode']}",
    )
    profiles = []
    metrics = []
    for diagonal_index, model_file in enumerate(CONFIG["model_files"]):
        print(f"Processing diagonal {diagonal_index}: {model_file}")
        profile, column_metrics = run_column(
            CONFIG,
            diagonal_index,
            model_file,
            deltas,
        )
        profiles.append(profile)
        metrics.append(column_metrics)
        print(
            f"true={column_metrics['true_response']:.6f} "
            f"wrong_mean={column_metrics['wrong_mean']:.6f} "
            f"separation={column_metrics['separation']:.4f}"
        )
        tf.keras.backend.clear_session()
    stacked = np.stack(profiles)
    npy_path = path("npy")
    delta_path = path("deltas.npy")
    json_path = path("json")
    np.save(npy_path, stacked)
    np.save(delta_path, deltas)
    write_json(
        json_path,
        {
            "config": CONFIG,
            "profile_file": os.path.basename(npy_path),
            "delta_file": os.path.basename(delta_path),
            "metrics": metrics,
        },
    )
    for diagonal_index, profile in enumerate(profiles):
        if CONFIG["save_png"]:
            plot_profile(
                profile,
                path(f"diag{diagonal_index}.png"),
                CONFIG,
                f"Byte column {column_flat_indices(diagonal_index)}",
            )
        if CONFIG["save_eps"]:
            plot_profile(
                profile,
                path(f"diag{diagonal_index}.eps"),
                CONFIG,
                f"Byte column {column_flat_indices(diagonal_index)}",
            )
    print(f"Profiles: {npy_path}")
    print(f"Deltas: {delta_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()
