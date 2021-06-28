import pandas as pd
import tensorflow as tf
import datetime
from tensorflow.keras import layers


ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


df = pd.read_csv("data/df.csv")
X_train = df['steps']
y_train = df['distance']

log_dir = "/tmp/tensorboard/{}".format(ts)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error',
    metrics='mean_absolute_error')

model.fit(
    X_train, y_train,
    epochs=13,
    callbacks=[tensorboard_callback]
)

print(model.summary())
model.save('models/{}'.format(ts))
