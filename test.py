import cv2
import numpy as np
from keras.models import load_model

model = load_model('model.h5')

labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
          13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

img = cv2.imread('./Dataset/test_set/A/1.png', 0)
img = cv2.resize(img, (64, 64))
img = np.array(img).reshape(-1, 64, 64, 1)


prediction = model.predict(img)
print(prediction)
predicted_letter = labels[np.argmax(prediction)]
print(predicted_letter)

# result = str(model.predict_step('./Dataset/test_set/A/1.png'))