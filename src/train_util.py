import tensorflow as tf

from util import prod


def limit_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus),
                  "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def train_loop(model, optimizer, loss_fn, train_ds, test_ds, epochs,
               batch_size, model_dir, train_log_dir, val_log_dir):
    """Execute a basic train loop and report train loss and accuracy
    to the train and val summary writers."""
    train_ds = train_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    # train and val metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = \
        tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
    val_loss = tf.metrics.Mean(name='val_loss')
    val_accuracy = \
        tf.keras.metrics.SparseCategoricalAccuracy(name='val_acc')
    val_summary_writer = tf.summary.create_file_writer(str(val_log_dir))

    # checkpoint and manager
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=1)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(y_true=labels, y_pred=predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
        train_accuracy(y_pred=predictions, y_true=labels)

    @tf.function
    def val_step(images, labels):
        predictions = model(images, training=False)
        loss = loss_fn(labels, predictions)
        val_loss(loss)
        val_accuracy(y_pred=predictions, y_true=labels)

    for epoch in range(1, epochs + 1):
        tf.print(f'Start of epoch {epoch}')
        with train_summary_writer.as_default():
            for batch in train_ds:
                train_step(batch['image'], batch['label'])
            # write metrics
            tf.summary.scalar('train loss', train_loss.result(), step=epoch)
            tf.summary.scalar('train acc', train_accuracy.result(), step=epoch)
            tf.print('train loss', train_loss.result())
            tf.print('train acc', train_accuracy.result())
            train_loss.reset_states()
            train_accuracy.reset_states()
        with val_summary_writer.as_default():
            for batch in test_ds:
                val_step(batch['image'], batch['label'])
            tf.summary.scalar('val loss', val_loss.result(), step=epoch)
            tf.summary.scalar('val acc', val_accuracy.result(), step=epoch)
            tf.print('val loss', val_loss.result())
            tf.print('val acc', val_accuracy.result())
            val_loss.reset_states()
            val_accuracy.reset_states()
        manager.save(epoch)


def calculate_trainable_variables(model):
    return sum(
        prod(variable.get_shape())
        for variable in model.trainable_variables
    )
