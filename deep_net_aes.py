import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import numpy as np
import time
from pickle import dump
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Conv1D, Input, Reshape, Permute, Add,
    BatchNormalization, Activation, GlobalAveragePooling2D, Concatenate
)
from tensorflow.keras.regularizers import l2
import AES_data_gen as data_gen


bs = 5000
wdir = os.path.join(_HERE, 'saved_models_simple')
os.makedirs(wdir, exist_ok=True)

_DEFAULT_NUM_BLOCKS = 8
_DEFAULT_WORD_SIZE = 16
_DEFAULT_BITS_PER_PAIR = _DEFAULT_NUM_BLOCKS * _DEFAULT_WORD_SIZE * 3

def get_dims(selected_bytes=None):
    if selected_bytes is None:
        return _DEFAULT_NUM_BLOCKS, _DEFAULT_WORD_SIZE, _DEFAULT_BITS_PER_PAIR, 16 * 3
    else:
        n_sel = len(selected_bytes)
        num_blocks = n_sel
        word_size = 8
        bits_per_pair = n_sel * 8 * 3
        bytes_per_pair = n_sel * 3
        return num_blocks, word_size, bits_per_pair, bytes_per_pair


def cyclic_lr(num_epochs, high_lr, low_lr):
    def schedule(i):
        return low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
    return schedule

def make_checkpoint(filepath):
    return ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True)


def make_resnet(group_size=2, num_blocks=_DEFAULT_NUM_BLOCKS, num_filters=32,
                num_outputs=1, word_size=_DEFAULT_WORD_SIZE, ks=3, depth=5,
                reg_param=0.0001, final_activation='sigmoid'):
    inp = Input(shape=(num_blocks * word_size * 3 * group_size,))
    rs = Reshape((group_size, 3 * num_blocks, word_size))(inp)
    perm = Permute((1, 3, 2))(rs)
    conv01 = Conv1D(num_filters, kernel_size=1, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv02 = Conv1D(num_filters, kernel_size=4, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv03 = Conv1D(num_filters, kernel_size=6, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    c2 = Concatenate(axis=-1)([conv01, conv02, conv03])
    conv0 = BatchNormalization()(c2)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0
    for i in range(depth):
        conv1 = Conv1D(num_filters * 3, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters * 3, kernel_size=ks,
                       padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
        ks += 2
    dense0 = GlobalAveragePooling2D()(shortcut)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(dense0)
    return Model(inputs=inp, outputs=out)

def auto_config(k, ram_gb=90):
    batch_size = 5000
    train_n = 10_000_000
    val_n = 1_000_000
    return {
        'max_lr': 0.002, 'min_lr': 0.0001,
        'batch_size': batch_size,
        'train_samples': train_n,
        'val_samples': val_n,
        'num_epochs': 20,
        'reg_param': 10**-5,
    }


def train_aes_distinguisher(num_rounds=3, diff=None,
                             group_size=2, depth=5, num_epochs=None,
                             train_samples=None, val_samples=None,
                             batch_size=None, max_lr=None, min_lr=None,
                             reg_param=None, auto=True, ram_gb=90,
                             selected_bytes=None):
    if diff is None:
        diff = [0x80] + [0x00] * 15
    
    num_blocks, word_size, bits_per_pair, bytes_per_pair = get_dims(selected_bytes)
    
    if auto:
        cfg = auto_config(group_size, ram_gb)
        if max_lr is None: max_lr = cfg['max_lr']
        if min_lr is None: min_lr = cfg['min_lr']
        if batch_size is None: batch_size = cfg['batch_size']
        if train_samples is None: train_samples = cfg['train_samples']
        if val_samples is None: val_samples = cfg['val_samples']
        if num_epochs is None: num_epochs = cfg['num_epochs']
        if reg_param is None: reg_param = cfg['reg_param']
    else:
        if max_lr is None: max_lr = 0.002
        if min_lr is None: min_lr = 0.0001
        if batch_size is None: batch_size = 5000
        if train_samples is None: train_samples = 10_000_000
        if val_samples is None: val_samples = 1_000_000
        if num_epochs is None: num_epochs = 20
        if reg_param is None: reg_param = 10**-5

    diff_arr = np.array(diff, dtype=np.uint8).flatten()
    nonzero_count = np.count_nonzero(diff_arr)
    if nonzero_count <= 4:
        diff_str = '_'.join(f'p{i}_{v:02x}' for i, v in enumerate(diff_arr) if v != 0)
    else:
        diff_str = 'x'.join(f'{v:02x}' for v in diff_arr)
    sel_str = "x".join(str(b) for b in selected_bytes) if selected_bytes else "full"
    tag = f"aes_r{num_rounds}_gs{group_size}_d{diff_str}_sel{sel_str}_d{depth}_invmc_simple"

    print("=" * 70)
    print(f"  selected_bytes = {selected_bytes}")
    print("=" * 70)
    print(f"  k={group_size}, nr={num_rounds}, depth={depth}")
    print(f"  训练集: {train_samples:.1e}, 验证集: {val_samples:.1e}")
    print(f"  batch={batch_size}, lr={max_lr:.0e}→{min_lr:.0e}")
    print("=" * 70)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        actual_batch = batch_size * strategy.num_replicas_in_sync
    else:
        strategy = tf.distribute.get_strategy()
        actual_batch = batch_size
    print(f"\n[GPU] {strategy.num_replicas_in_sync} devices, batch={actual_batch}")

    with strategy.scope():
        net = make_resnet(group_size=group_size, num_blocks=num_blocks,
                          word_size=word_size, depth=depth, reg_param=reg_param)
        net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    print(f"[模型] 参数量: {net.count_params():,}")

    print(f"\n{'─' * 50}")
    print(f"  阶段 1/2: 生成训练集")
    print(f"{'─' * 50}")
    t0 = time.time()
    X_train, Y_train = data_gen.make_dataset_with_group_size(
        train_samples, num_rounds, diff=diff, group_size=group_size,
        selected_bytes=selected_bytes,
    )
    Y_train = Y_train.astype(np.float32)
    print(f"  训练集 shape: {X_train.shape}, 耗时: {time.time()-t0:.1f}s")

    print(f"\n{'─' * 50}")
    print(f"  阶段 2/2: 生成验证集")
    print(f"{'─' * 50}")
    t0 = time.time()
    X_val, Y_val = data_gen.make_dataset_with_group_size(
        val_samples, num_rounds, diff=diff, group_size=group_size,
        selected_bytes=selected_bytes,
    )
    Y_val = Y_val.astype(np.float32)
    print(f"  验证集 shape: {X_val.shape}, 耗时: {time.time()-t0:.1f}s")

    print(f"\n{'─' * 50}")
    print(f"  开始训练 ({num_epochs} epochs)")
    print(f"{'─' * 50}")
    check_path = os.path.join(wdir, f'{tag}_best.h5')
    check = make_checkpoint(check_path)
    lr = LearningRateScheduler(cyclic_lr(10, max_lr, min_lr))
    
    t0 = time.time()
    h = net.fit(
        X_train, Y_train,
        epochs=num_epochs,
        batch_size=actual_batch,
        validation_data=(X_val, Y_val),
        callbacks=[lr, check],
        verbose=2,
    )
    total_t = time.time() - t0

    best_epoch = int(np.argmin(h.history['val_loss']))
    best_vl = h.history['val_loss'][best_epoch]
    best_va = h.history['val_acc'][best_epoch]
    final_va = h.history['val_acc'][-1]

    print(f"\n{'=' * 70}")
    print(f"  最低 val_loss: {best_vl:.6f} (epoch {best_epoch + 1})")
    print(f"  该 epoch 的 val_acc: {best_va:.4f}  |  最终 val_acc: {final_va:.4f}")
    print(f"  训练时间: {total_t:.1f}s ({total_t/60:.1f}min)")
    print(f"{'=' * 70}")

    best_h5 = f'{tag}_best_{best_va:.4f}.h5'
    if os.path.exists(check_path):
        os.rename(check_path, os.path.join(wdir, best_h5))
    print(f"  最佳模型: {best_h5}")

    hist_path = os.path.join(wdir, f'{tag}_hist_{best_va:.4f}.p')
    dump(h.history, open(hist_path, 'wb'))
    print(f"  训练历史: {hist_path}")

    return net, h

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)


    SELECTED = [0, 1, 2, 3]   # 选择的字节索引
    rounds = [3]
    group_sizes = [2, 4, 8, 16, 32]
    diff = [0x80] + [0x00] * 15

    for r in rounds:
        for gs in group_sizes:
            print(f"\n{'='*60}")
            print(f"AES r={r}, k={gs}, sel={SELECTED}, mode=N+1 peel + InvMC")
            print(f"{'='*60}")
            train_aes_distinguisher(
                num_rounds=r,
                diff=diff,
                group_size=gs,
                depth=5,
                auto=True,
                ram_gb=90,
                selected_bytes=SELECTED,
            )
