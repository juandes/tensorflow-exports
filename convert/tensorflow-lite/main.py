import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(
    '../../models/steps-regression/saved-model')
tflite_model = converter.convert()

# Save the model.
with open('../../models/steps-regression/tflite/modellll.tflite', 'wb') as f:
    f.write(tflite_model)


# With quantization
# Floating point -> integer
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

with open('../../models/steps-regression/tflite/model_int_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)


# Float16 quant
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()

with open('../../models/steps-regression/tflite/model_float16_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)
