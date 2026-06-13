import tensorflow as tf
import numpy as np
import os

# Load model
model = tf.keras.models.load_model("model/crop_disease_model.h5")

IMG_SIZE = (224, 224)

# Test image path
img_path = r"dataset\Test\Tomato - Healthy\085cbe78-1d5c-45eb-877f-f409526032d5___GH_HL Leaf 469.JPG"

# Load image
img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)

# Convert to array
img_array = tf.keras.utils.img_to_array(img)

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Get class names automatically from Train folder
class_names = sorted([
    d for d in os.listdir("dataset/Train")
    if os.path.isdir(os.path.join("dataset/Train", d))
])

# Get prediction
pred_index = np.argmax(prediction)
confidence = np.max(prediction) * 100

# Print result
print("\n==========================")
print("Disease :", class_names[pred_index])
print("Confidence :", round(float(confidence), 2), "%")
print("==========================")