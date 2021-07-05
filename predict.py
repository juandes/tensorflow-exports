import tensorflow as tf

model = tf.keras.models.load_model('models/steps-regression/saved-model')
print(model.summary())
print(model.predict([6000]))
