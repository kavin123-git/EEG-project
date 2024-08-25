from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

from google.colab import files

uploaded = files.upload()

img_path = 'tomato.jpeg'

# Load the image
img = image.load_img(img_path, target_size=(224, 224))

# Convert the image to an array
x = image.img_to_array(img)

# Expand dimensions to match the input shape expected by the model
x = np.expand_dims(x, axis=0)

# Preprocess the input image
x = preprocess_input(x)

# Load the pretrained ResNet50 model
model = ResNet50(weights='imagenet')

# Predict the class of the image
preds = model.predict(x)

# Decode and print the predictions
print('Predicted:', decode_predictions(preds, top=3)[0])
