import json
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import aes_cipher as aes
from train_distinguisher import get_dims, make_resnet


HERE = os.path.dirname(os.path.abspath(__file__))


def configure_tensorflow():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return len(gpus)


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(HERE, path)


def make_output_paths(output_dir, prefix):
    directory = resolve_path(output_dir) if output_dir else HERE
    os.makedirs(directory, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def path(extension):
        return os.path.join(directory, f"{prefix}_{stamp}.{extension}")

    return path


def load_distinguisher(model_file, group_size, selected_bytes, depth, reg_param=1e-5):
    num_blocks, word_size, _ = get_dims(selected_bytes)
    model = make_resnet(
        group_size=group_size,
        num_blocks=num_blocks,
        word_size=word_size,
        depth=depth,
        reg_param=reg_param,
    )
    model.compile(optimizer="adam", loss="mse", metrics=["acc"])
    model_path = resolve_path(model_file)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")
    model.load_weights(model_path)
    return model


def diagonal_indices(diagonal_index):
    if diagonal_index not in range(4):
        raise ValueError("diagonal_index must be 0, 1, 2, or 3")
    return sorted(
        4 * column + row
        for column in range(4)
        for row in range(4)
        if (column + row) % 4 == diagonal_index
    )


def column_flat_indices(column_index):
    if column_index not in range(4):
        raise ValueError("column_index must be 0, 1, 2, or 3")
    return [4 * column_index + row for row in range(4)]


def peel_one_column_invmc_batch(ciphertexts, key_candidates, column_index):
    order = np.asarray(
        [4 * ((column_index - row) % 4) + row for row in range(4)],
        dtype=np.intp,
    )
    diagonal = ciphertexts[None, :, order] ^ key_candidates[:, None, order]
    after_sub_bytes = aes.INV_SBOX[diagonal]
    peeled = np.zeros_like(after_sub_bytes)
    for row in range(4):
        for source_row in range(4):
            peeled[..., row] ^= aes._galois_lookup[
                aes._INV_MIX_MAT[row, source_row],
                after_sub_bytes[..., source_row],
            ]
    return peeled


def bytes_to_bits_batch(first, second):
    candidate_count, pair_count, byte_count = first.shape
    values = np.concatenate((first, second, first ^ second), axis=-1)
    values = values.reshape(candidate_count * pair_count, byte_count * 3)
    bits = np.zeros(
        (candidate_count * pair_count, byte_count * 3 * 8),
        dtype=np.uint8,
    )
    for byte_index in range(byte_count * 3):
        for bit_index in range(8):
            bits[:, byte_index * 8 + bit_index] = (
                values[:, byte_index] >> (7 - bit_index)
            ) & 1
    return bits.reshape(candidate_count, pair_count, byte_count * 24)


def generate_attack_data(config, seed_offset=0):
    rng = np.random.RandomState(config["data_seed"] + seed_offset)
    rounds = config["attack_rounds"]
    pair_count = config["group_size"] * config["max_instances"]
    master_key = rng.randint(0, 256, size=(1, 16), dtype=np.uint8)
    one_schedule = aes.expand_key(master_key, rounds)
    last_round_key = one_schedule[-1, 0].copy()
    schedule = np.tile(one_schedule, (1, pair_count, 1))
    plaintext0 = rng.randint(0, 256, size=(pair_count, 16), dtype=np.uint8)
    difference = np.asarray(config["diff"], dtype=np.uint8).reshape(1, 16)
    plaintext1 = plaintext0 ^ difference
    ciphertext0 = aes.encrypt(
        plaintext0,
        schedule,
        nr=rounds,
        final_mc=False,
    )
    ciphertext1 = aes.encrypt(
        plaintext1,
        schedule,
        nr=rounds,
        final_mc=False,
    )
    return {
        "master_key": master_key[0],
        "last_round_key": last_round_key,
        "plaintext0": plaintext0,
        "plaintext1": plaintext1,
        "ciphertext0": ciphertext0,
        "ciphertext1": ciphertext1,
    }


def generate_deltas(wrong_key_count, mode, seed):
    if wrong_key_count < 1 or wrong_key_count >= 2**32:
        raise ValueError("wrong_key_count must satisfy 1 <= count < 2**32")
    if mode == "contiguous":
        values = np.arange(wrong_key_count + 1, dtype=np.uint64)
    elif mode == "random":
        rng = np.random.RandomState(seed)
        chosen = set()
        while len(chosen) < wrong_key_count:
            remaining = wrong_key_count - len(chosen)
            batch = rng.randint(
                1,
                2**32,
                size=max(1024, remaining * 2),
                dtype=np.uint64,
            )
            chosen.update(int(value) for value in batch)
        values = np.empty(wrong_key_count + 1, dtype=np.uint64)
        values[0] = 0
        values[1:] = np.fromiter(
            list(chosen)[:wrong_key_count],
            dtype=np.uint64,
            count=wrong_key_count,
        )
        rng.shuffle(values[1:])
    else:
        raise ValueError("mode must be 'contiguous' or 'random'")
    deltas = np.empty((wrong_key_count + 1, 4), dtype=np.uint8)
    for byte_index, shift in enumerate((24, 16, 8, 0)):
        deltas[:, byte_index] = (values >> shift) & 0xFF
    return deltas


def candidates_from_deltas(last_round_key, diagonal_index, deltas):
    indices = diagonal_indices(diagonal_index)
    candidates = np.tile(last_round_key, (len(deltas), 1))
    candidates[:, indices] ^= deltas
    return candidates


def score_candidates_for_counts(
    model,
    ciphertext0,
    ciphertext1,
    candidates,
    diagonal_index,
    group_size,
    instance_counts,
    candidate_batch_size,
    prediction_batch_size,
    progress_label="",
):
    counts = sorted(set(int(value) for value in instance_counts))
    if not counts or counts[0] < 1:
        raise ValueError("instance_counts must contain positive integers")
    max_instances = counts[-1]
    expected_pairs = group_size * max_instances
    if len(ciphertext0) != expected_pairs or len(ciphertext1) != expected_pairs:
        raise ValueError("ciphertext count does not match group_size * max_instances")
    candidate_count = len(candidates)
    dimension = group_size * 4 * 3 * 8
    profiles = np.empty((len(counts), candidate_count), dtype=np.float64)
    started = time.time()
    for start in range(0, candidate_count, candidate_batch_size):
        end = min(start + candidate_batch_size, candidate_count)
        candidate_batch = candidates[start:end]
        batch_count = end - start
        peeled0 = peel_one_column_invmc_batch(
            ciphertext0,
            candidate_batch,
            diagonal_index,
        )
        peeled1 = peel_one_column_invmc_batch(
            ciphertext1,
            candidate_batch,
            diagonal_index,
        )
        bits = bytes_to_bits_batch(peeled0, peeled1)
        grouped = bits.reshape(batch_count, max_instances, dimension)
        flattened = grouped.reshape(batch_count * max_instances, dimension)
        predictions = model.predict(
            flattened.astype(np.float32),
            batch_size=prediction_batch_size,
            verbose=0,
        ).reshape(batch_count, max_instances)
        cumulative = np.cumsum(predictions, axis=1, dtype=np.float64)
        for row, count in enumerate(counts):
            profiles[row, start:end] = cumulative[:, count - 1] / count
        elapsed = time.time() - started
        eta = elapsed / end * (candidate_count - end) if end else 0.0
        suffix = f" {progress_label}" if progress_label else ""
        print(
            f"[{end:,}/{candidate_count:,}]{suffix} "
            f"elapsed={elapsed:.1f}s ETA={eta:.1f}s",
            flush=True,
        )
    return counts, profiles, time.time() - started


def response_metrics(profile, true_index):
    values = np.asarray(profile, dtype=np.float64)
    if true_index < 0 or true_index >= len(values):
        raise IndexError("true_index is outside the response profile")
    wrong_values = np.delete(values, true_index)
    true_response = float(values[true_index])
    wrong_mean = float(np.mean(wrong_values))
    wrong_std = float(np.std(wrong_values))
    wrong_max = float(np.max(wrong_values))
    predicted_index = int(np.argmax(values))
    separation = (
        (true_response - wrong_mean) / wrong_std
        if wrong_std > 0
        else float("inf")
    )
    return {
        "true_response": true_response,
        "wrong_mean": wrong_mean,
        "wrong_std": wrong_std,
        "wrong_max": wrong_max,
        "response_margin": true_response - wrong_max,
        "separation": separation,
        "predicted_index": predicted_index,
        "correct_selected": predicted_index == int(true_index),
    }


def enumerate_restricted_diagonal(true_diagonal, unknown_bit_positions):
    positions = [int(position) for position in unknown_bit_positions]
    if len(positions) != len(set(positions)):
        raise ValueError("unknown bit positions must be unique")
    if any(position < 0 or position > 31 for position in positions):
        raise ValueError("unknown bit positions must be in [0, 31]")
    if len(positions) > 24:
        raise ValueError("at most 24 unknown bits may be materialized")
    true_value = np.asarray(true_diagonal, dtype=np.uint8).reshape(4)
    base = true_value.copy()
    for position in positions:
        byte_index, bit_in_byte = divmod(position, 8)
        base[byte_index] &= np.uint8(~(1 << (7 - bit_in_byte)) & 0xFF)
    candidate_count = 1 << len(positions)
    values = np.arange(candidate_count, dtype=np.uint32)
    candidates = np.tile(base, (candidate_count, 1))
    true_index = 0
    for source_bit, position in enumerate(positions):
        byte_index, bit_in_byte = divmod(position, 8)
        mask = np.uint8(1 << (7 - bit_in_byte))
        candidates[:, byte_index] |= (
            ((values >> source_bit) & 1).astype(np.uint8) * mask
        )
        true_index |= int((true_value[byte_index] & mask) != 0) << source_bit
    if not np.array_equal(candidates[true_index], true_value):
        raise AssertionError("restricted enumeration omitted the true diagonal")
    return candidates, true_index


def full_key_candidates(diagonal_values, diagonal_index):
    candidates = np.zeros((len(diagonal_values), 16), dtype=np.uint8)
    candidates[:, diagonal_indices(diagonal_index)] = diagonal_values
    return candidates


def invert_aes128_round_key(round_key, round_number):
    if round_number < 1 or round_number >= len(aes.RCON):
        raise ValueError("round_number is outside the AES-128 key schedule")
    current = np.asarray(round_key, dtype=np.uint8).reshape(4, 4).T.copy()
    for round_index in range(round_number, 0, -1):
        previous = np.empty_like(current)
        previous[3] = current[3] ^ current[2]
        previous[2] = current[2] ^ current[1]
        previous[1] = current[1] ^ current[0]
        transformed = np.roll(previous[3], -1)
        transformed = aes.SBOX[transformed]
        transformed[0] ^= aes.RCON[round_index]
        previous[0] = current[0] ^ transformed
        current = previous
    return current.T.reshape(16)


def encrypt_with_one_key(plaintexts, key, rounds):
    one_schedule = aes.expand_key(
        np.asarray(key, dtype=np.uint8).reshape(1, 16),
        rounds,
    )
    schedule = np.tile(one_schedule, (1, len(plaintexts), 1))
    return aes.encrypt(plaintexts, schedule, nr=rounds, final_mc=False)


def key_hex(key):
    return "".join(f"{int(byte):02x}" for byte in np.asarray(key).reshape(-1))


def write_json(path, payload):
    def convert(value):
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            convert(payload),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
