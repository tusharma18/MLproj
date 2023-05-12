from keras.models import load_model

model = load_model('model.h5')

result = model.predict_step(1, './Dataset/test_set/A/1.png')