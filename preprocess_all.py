import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# ---------------------------
# Paths
# --------------------------
train_dir = r"C:\Users\asus\OneDrive\Desktop\skin dectation project\LungcancerDataSet\Data\train"
val_dir   = r"C:\Users\asus\OneDrive\Desktop\skin dectation project\LungcancerDataSet\Data\valid"
test_dir  = r"C:\Users\asus\OneDrive\Desktop\skin dectation project\LungcancerDataSet\Data\test"

# ---------------------------
# Image Preprocessing
# ---------------------------
img_size = (224, 224)   # You can use 128x128 or 224x224
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_gen = datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# ---------------------------
# Model Definition
# ---------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ---------------------------
# Train Model
# ---------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# ---------------------------
# Save Model as .h5
# ---------------------------
model.save("lung_cancer_model.h5")

print("✅ Model saved as lung_cancer_model.h5")
