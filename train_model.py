# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import h5py

img_size = 32

print("Loading the data...")
# 准备好的数据集
hf = h5py.File('/home/eyue/graduation_desigh_QinJiang/my_dataset/concrete_crack_image_data.h5', 'r')
X = np.array(hf.get('X_concrete'))
y = np.array(hf.get("y_concrete"))
hf.close()
print("Data successfully loaded!")

print("Scaling the data...!")
X = X / 255
print("Data successfully scaled!")

model = Sequential()

model.add(Conv2D(16, (5, 5), activation = "relu", input_shape = (img_size, img_size, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.4))

model.add(Conv2D(32, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(.3))

model.add(Flatten())
model.add(Dense(258, activation = "relu"))

model.add(Dense(1, activation = "sigmoid"))

print("Compiling the model...")
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
print("Model successfully compiled!!")

print("Fitting the model...")
model.fit(X, y, batch_size = 64, epochs = 32, validation_split = .1)
print("Model successfully fitted!!")

print("Saving the model...")
# 训练好的模型
model.save("/home/eyue/graduation_desigh_QinJiang/my_model/Concrete_Crack_Classification_model.model")
print("Model successfully saved!!")
