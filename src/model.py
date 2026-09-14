"""TV1 — ResNetv1-6 1D điều chỉnh từ Hình 4 của Novac et al. (2021)."""


def build_model(config):
    """Tạo ResNetv1-6 với Conv/Pool 1D như mô tả ở mục 6 của paper."""
    import tensorflow as tf

    input_shape = tuple(config["dataset"]["input_shape"])
    filters = int(config["model"]["filters"])
    class_count = len(config["dataset"]["labels"])
    if input_shape != (128, 9):
        raise ValueError(f"UCI-HAR yêu cầu input_shape [128, 9], nhận {input_shape}")
    if filters <= 0:
        raise ValueError("Số filters phải dương")

    inputs = tf.keras.Input(shape=input_shape, name="sensor_window")

    # Residual block 1: projection shortcut + temporal downsampling.
    main = tf.keras.layers.Conv1D(filters, 3, padding="same", name="block1_conv1")(
        inputs
    )
    main = tf.keras.layers.ReLU(name="block1_relu1")(main)
    main = tf.keras.layers.Conv1D(filters, 3, padding="same", name="block1_conv2")(
        main
    )
    main = tf.keras.layers.MaxPooling1D(2, name="block1_main_pool")(main)
    shortcut = tf.keras.layers.Conv1D(
        filters, 1, padding="valid", name="block1_projection"
    )(inputs)
    shortcut = tf.keras.layers.MaxPooling1D(2, name="block1_shortcut_pool")(
        shortcut
    )
    x = tf.keras.layers.Add(name="block1_add")([main, shortcut])
    x = tf.keras.layers.ReLU(name="block1_output")(x)

    # Residual block 2: identity shortcut, no change in temporal resolution.
    shortcut = x
    main = tf.keras.layers.Conv1D(filters, 3, padding="same", name="block2_conv1")(x)
    main = tf.keras.layers.ReLU(name="block2_relu1")(main)
    main = tf.keras.layers.Conv1D(filters, 3, padding="same", name="block2_conv2")(
        main
    )
    x = tf.keras.layers.Add(name="block2_add")([main, shortcut])
    x = tf.keras.layers.ReLU(name="block2_output")(x)

    x = tf.keras.layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    outputs = tf.keras.layers.Dense(class_count, name="activity_logits")(x)
    return tf.keras.Model(inputs, outputs, name="uci_har_resnetv1_6_fp32")
