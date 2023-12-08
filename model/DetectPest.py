from keras.models import load_model
import os, sys
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

from keras.models import load_model
import os, sys
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np

model = load_model('model/model.h5')
def preprocess_image_for_prediction(img_path):
    print(img_path)
    scale_size = 112
    img = plt.imread(img_path)
    img = cv2.resize(img, (scale_size, scale_size))

    # Handle images with alpha channel
    if img.shape == (scale_size, scale_size, 4):
        img = img[:, :, :3]

    # Normalize pixel values to the range [0, 1]
    img = img / 255.0

    # Expand dimensions to match the model's expected input shape
    img = np.expand_dims(img, axis=0)

    return img

def get_pest_prediction(imagepath):
    pests = ["Jute stem girdler", "Jute hairy", "Field Cricket", "Beet spot flies", "Ampelophaga"]
    image = preprocess_image_for_prediction(imagepath)
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions[0])
    return pests[predicted_class_index]

