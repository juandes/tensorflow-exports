import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(
    '../../models/steps-regression/saved-model')
tflite_model = converter.convert()

# Save the model.
with open('../../models/steps-regression/tflite/model.tflite', 'wb') as f:
    f.write(tflite_model)
