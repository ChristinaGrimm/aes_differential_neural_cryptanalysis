import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
from os import urandom
import aes_cipher as aes


def _peel_one_round_invmc(ct, k4):
    n = ct.shape[0]
    x = ct ^ k4
    x_state = x.reshape(n, 4, 4).transpose(0, 2, 1)
    y_state = aes._inv_shift_rows(x_state)
    z_state = aes._inv_sub_bytes(y_state)
    invmc_state = aes._inv_mix_columns(z_state)
    return invmc_state.transpose(0, 2, 1).reshape(n, 16)


def _generate_positive_samples(n, nr, diff, selected_bytes):
    n_encrypt = nr + 1
    diff_arr = np.array(diff, dtype=np.uint8).reshape(1, 16)
    pt0 = np.frombuffer(urandom(16 * n), dtype=np.uint8).reshape(-1, 16)
    pt1 = pt0 ^ diff_arr
    keys = np.frombuffer(urandom(16 * n), dtype=np.uint8).reshape(-1, 16)
    ks = aes.expand_key(keys, n_encrypt)
    true_k4 = ks[-1].copy()
    ct0 = aes.encrypt(pt0, ks, nr=n_encrypt, final_mc=False)
    ct1 = aes.encrypt(pt1, ks, nr=n_encrypt, final_mc=False)
    c0 = _peel_one_round_invmc(ct0, true_k4)
    c1 = _peel_one_round_invmc(ct1, true_k4)
    return _bytes_to_bits(c0, c1, selected_bytes)


def _generate_negative_samples(n, nr, diff, selected_bytes):
    n_encrypt = nr + 1
    diff_arr = np.array(diff, dtype=np.uint8).reshape(1, 16)
    pt0 = np.frombuffer(urandom(16 * n), dtype=np.uint8).reshape(-1, 16)
    pt1 = pt0 ^ diff_arr
    keys = np.frombuffer(urandom(16 * n), dtype=np.uint8).reshape(-1, 16)
    ks = aes.expand_key(keys, n_encrypt)
    true_k4 = ks[-1].copy()
    ct0 = aes.encrypt(pt0, ks, nr=n_encrypt, final_mc=False)
    ct1 = aes.encrypt(pt1, ks, nr=n_encrypt, final_mc=False)
    wrong_k4 = true_k4 ^ np.random.randint(0, 256, size=(n, 16), dtype=np.uint8)
    for i in range(n):
        if np.array_equal(wrong_k4[i], true_k4[i]):
            wrong_k4[i, 0] ^= 0x01
    c0 = _peel_one_round_invmc(ct0, wrong_k4)
    c1 = _peel_one_round_invmc(ct1, wrong_k4)
    return _bytes_to_bits(c0, c1, selected_bytes)


def _bytes_to_bits(c0, c1, selected_bytes):
    N_BYTES_PER_CT = 16
    c_diff = c0 ^ c1
    X = np.concatenate((c0, c1, c_diff), axis=1)
    if selected_bytes is not None:
        indices = []
        for offset in [0, N_BYTES_PER_CT, 2 * N_BYTES_PER_CT]:
            for pos in selected_bytes:
                indices.append(offset + pos)
        X = X[:, indices]
    n_bytes = X.shape[1]
    X_bits = np.zeros((X.shape[0], n_bytes * 8), dtype=np.uint8)
    for i in range(n_bytes):
        for b in range(8):
            X_bits[:, i * 8 + b] = (X[:, i] >> (7 - b)) & 1
    return X_bits


def make_target_diff_samples(n=10**7, nr=3, diff_type=1, diff=None, selected_bytes=None):
    if diff is None:
        diff = [0x80] + [0x00] * 15
    if diff_type == 1:
        X_bits = _generate_positive_samples(n, nr, diff, selected_bytes)
    else:
        X_bits = _generate_negative_samples(n, nr, diff, selected_bytes)
    return X_bits

def make_dataset_with_group_size(n, nr, diff=None, group_size=2, selected_bytes=None):
    if diff is None:
        diff = [0x80] + [0x00] * 15
    assert n % group_size == 0
    num = n // 2
    X_p = _generate_positive_samples(num, nr, diff, selected_bytes)
    X_n = _generate_negative_samples(num, nr, diff, selected_bytes)
    X_raw = np.concatenate((X_p, X_n), axis=0)
    n_raw, m = np.shape(X_raw)
    X = X_raw.reshape(-1, group_size * m)
    Y_p = np.ones(num // group_size, dtype=np.uint8)
    Y_n = np.zeros(num // group_size, dtype=np.uint8)
    Y = np.concatenate((Y_p, Y_n))
    return X, Y


if __name__ == "__main__":
    import time
    sel = [0, 1, 2, 3]
    n_test, nr_test, gs_test = 1000, 3, 2
    print("=" * 60)
    print("对抗式负样本: N+1 方案 (对称版)")
    print(f"  selected_bytes={sel}, k={gs_test}, n={n_test}")
    print("=" * 60)
    t0 = time.time()
    X, Y = make_dataset_with_group_size(n_test, nr_test,
                                         diff=[0x80] + [0x00] * 15,
                                         group_size=gs_test,
                                         selected_bytes=sel)
    t = time.time() - t0
    print(f"  X: {X.shape}, Y: {Y.shape}")
    print(f"  正样本: {int(Y.sum())}, 负样本: {len(Y)-int(Y.sum())}")
    print(f"  耗时: {t:.3f}s")
    assert np.all((X == 0) | (X == 1)), "非二值!"
    print("  ✓ 验证通过")
