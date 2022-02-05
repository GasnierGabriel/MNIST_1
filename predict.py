import cv2
import keras
import numpy as np

nomImage = "3.jpg"
image = cv2.imread("images/" + nomImage, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (28, 28) )
image = np.asarray(image)
image = image.reshape(1, 28*28)
image = image.astype("float32") / 255

model = keras.models.load_model("model1.h5")

prediction = model.predict(image)
prediction = np.argmax(prediction)

print("l'image :" + nomImage + "est un :" + str(prediction))