from tensorflow.keras.models import load_model
model = load_model('mnist_cnn_model.h5')

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


img_path = 'testimage.png'  # path to image
img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))

img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = img_array.reshape(1, 28, 28, 1)

plt.imshow(img_array.reshape(28, 28))
plt.show()


predictions = model.predict(img_array)
predicted_digit = np.argmax(predictions)
print(f"Predicted Digit: {predicted_digit}")
