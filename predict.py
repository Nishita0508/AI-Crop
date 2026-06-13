import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model/crop_disease_model.h5")

IMG_SIZE = (224,224)

img_path = "test.jpg"  # change image name here

img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Tomato___Late_blight"
    # (we will fix full list next)
]

print("Prediction:", class_names[np.argmax(prediction)])