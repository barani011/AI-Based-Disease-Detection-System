import tensorflow as tf

model = tf.keras.models.load_model("lung_cancer_classifier.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 🚨 Important: Only use old built-in ops
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

# Optional: shrink model
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("lung_cancer_model_legacy.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Exported: lung_cancer_model_legacy.tflite")
