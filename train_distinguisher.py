import os
import time
from pickle import dump

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    Permute,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

import AES_data_gen as data_gen


HERE = os.path.dirname(os.path.abspath(__file__))


CONFIG = {
    # Distinguisher and data parameters
    "num_rounds": 3,
    "diff": [0x80] + [0x00] * 15,
    "group_sizes": [2, 4, 8, 16, 32],
    "selected_bytes": [0, 1, 2, 3],
    "depth": 5,
    # Training parameters
    "train_pairs": 10_000_000,
    "validation_pairs": 1_000_000,
    "epochs": 20,
    "batch_size_per_device": 5000,
    "max_lr": 2e-3,
    "min_lr": 1e-4,
    "lr_cycle": 10,
    "reg_param": 1e-5,
    # Output parameters
    "output_dir": "saved_models",
    "model_prefix": "method2",
}


def get_dims(selected_bytes=None):
    if selected_bytes is None:
        return 8, 16, 384
    n_selected = len(selected_bytes)
    return n_selected, 8, n_selected * 24


def cyclic_lr(period, high_lr, low_lr):
    if period < 2:
        raise ValueError("period must be at least 2")

    def schedule(epoch):
        position = epoch % period
        return low_lr + (period - 1 - position) / (period - 1) * (high_lr - low_lr)

    return schedule


def make_resnet(
    group_size=2,
    num_blocks=8,
    num_filters=32,
    num_outputs=1,
    word_size=16,
    kernel_size=3,
    depth=5,
    reg_param=1e-5,
    final_activation="sigmoid",
):
    inp = Input(shape=(num_blocks * word_size * 3 * group_size,))
    reshaped = Reshape((group_size, 3 * num_blocks, word_size))(inp)
    permuted = Permute((1, 3, 2))(reshaped)
    conv1 = Conv1D(
        num_filters,
        kernel_size=1,
        padding="same",
        kernel_regularizer=l2(reg_param),
    )(permuted)
    conv4 = Conv1D(
        num_filters,
        kernel_size=4,
        padding="same",
        kernel_regularizer=l2(reg_param),
    )(permuted)
    conv6 = Conv1D(
        num_filters,
        kernel_size=6,
        padding="same",
        kernel_regularizer=l2(reg_param),
    )(permuted)
    shortcut = Concatenate(axis=-1)([conv1, conv4, conv6])
    shortcut = BatchNormalization()(shortcut)
    shortcut = Activation("relu")(shortcut)
    current_kernel = kernel_size
    for _ in range(depth):
        residual = Conv1D(
            num_filters * 3,
            kernel_size=current_kernel,
            padding="same",
            kernel_regularizer=l2(reg_param),
        )(shortcut)
        residual = BatchNormalization()(residual)
        residual = Activation("relu")(residual)
        residual = Conv1D(
            num_filters * 3,
            kernel_size=current_kernel,
            padding="same",
            kernel_regularizer=l2(reg_param),
        )(residual)
        residual = BatchNormalization()(residual)
        residual = Activation("relu")(residual)
        shortcut = Add()([shortcut, residual])
        current_kernel += 2
    pooled = GlobalAveragePooling2D()(shortcut)
    out = Dense(
        num_outputs,
        activation=final_activation,
        kernel_regularizer=l2(reg_param),
    )(pooled)
    return Model(inputs=inp, outputs=out)


def configure_tensorflow():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy(), len(gpus)
    return tf.distribute.get_strategy(), max(1, len(gpus))


def validate_config(config, group_size):
    if group_size < 1:
        raise ValueError("group_size must be positive")
    if len(config["diff"]) != 16:
        raise ValueError("diff must contain 16 bytes")
    selected = config["selected_bytes"]
    if selected is not None:
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("selected_bytes must contain unique byte indices")
        if any(index < 0 or index > 15 for index in selected):
            raise ValueError("selected byte indices must be in [0, 15]")
    divisor = 2 * group_size
    if config["train_pairs"] % divisor != 0:
        raise ValueError("train_pairs must be divisible by 2 * group_size")
    if config["validation_pairs"] % divisor != 0:
        raise ValueError("validation_pairs must be divisible by 2 * group_size")


def experiment_tag(config, group_size):
    diff = np.asarray(config["diff"], dtype=np.uint8)
    diff_tag = "_".join(
        f"p{index}_{int(value):02x}"
        for index, value in enumerate(diff)
        if value != 0
    )
    selected = config["selected_bytes"]
    selected_tag = "full" if selected is None else "x".join(map(str, selected))
    return (
        f"aes_r{config['num_rounds']}_gs{group_size}_d{diff_tag}_"
        f"sel{selected_tag}_d{config['depth']}_{config['model_prefix']}"
    )


def train_one(config, group_size, strategy, device_count):
    validate_config(config, group_size)
    num_blocks, word_size, _ = get_dims(config["selected_bytes"])
    output_dir = config["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(HERE, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    tag = experiment_tag(config, group_size)
    checkpoint_path = os.path.join(output_dir, f"{tag}_best.h5")
    global_batch_size = config["batch_size_per_device"] * device_count

    with strategy.scope():
        model = make_resnet(
            group_size=group_size,
            num_blocks=num_blocks,
            word_size=word_size,
            depth=config["depth"],
            reg_param=config["reg_param"],
        )
        model.compile(optimizer="adam", loss="mse", metrics=["acc"])

    started = time.time()
    x_train, y_train = data_gen.make_dataset_with_group_size(
        config["train_pairs"],
        config["num_rounds"],
        diff=config["diff"],
        group_size=group_size,
        selected_bytes=config["selected_bytes"],
    )
    x_validation, y_validation = data_gen.make_dataset_with_group_size(
        config["validation_pairs"],
        config["num_rounds"],
        diff=config["diff"],
        group_size=group_size,
        selected_bytes=config["selected_bytes"],
    )
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        ),
        LearningRateScheduler(
            cyclic_lr(config["lr_cycle"], config["max_lr"], config["min_lr"])
        ),
    ]
    history = model.fit(
        x_train,
        y_train.astype(np.float32),
        epochs=config["epochs"],
        batch_size=global_batch_size,
        validation_data=(x_validation, y_validation.astype(np.float32)),
        callbacks=callbacks,
        verbose=2,
    )
    best_epoch = int(np.argmin(history.history["val_loss"]))
    best_accuracy = float(history.history["val_acc"][best_epoch])
    final_model_path = os.path.join(
        output_dir,
        f"{tag}_best_{best_accuracy:.4f}.h5",
    )
    if os.path.exists(checkpoint_path):
        os.replace(checkpoint_path, final_model_path)
    history_path = os.path.join(
        output_dir,
        f"{tag}_history_{best_accuracy:.4f}.p",
    )
    with open(history_path, "wb") as handle:
        dump(history.history, handle)
    return {
        "model": model,
        "history": history,
        "model_path": final_model_path,
        "history_path": history_path,
        "best_epoch": best_epoch + 1,
        "best_accuracy": best_accuracy,
        "seconds": time.time() - started,
    }


def main():
    # Edit CONFIG before running a full experiment.
    strategy, device_count = configure_tensorflow()
    print(f"TensorFlow devices used for batching: {device_count}")
    for group_size in CONFIG["group_sizes"]:
        print("=" * 72)
        print(
            f"Method 2 training: rounds={CONFIG['num_rounds']}, "
            f"m={group_size}, selected_bytes={CONFIG['selected_bytes']}"
        )
        result = train_one(CONFIG, group_size, strategy, device_count)
        print(f"Best validation accuracy: {result['best_accuracy']:.4f}")
        print(f"Best epoch: {result['best_epoch']}")
        print(f"Model: {result['model_path']}")
        print(f"History: {result['history_path']}")
        print(f"Elapsed: {result['seconds']:.1f} seconds")
        tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
