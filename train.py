import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.layers import BatchNormalization

def loadImageFile(fileimage):
    f = open(fileimage, "rb")
    f.read(16)
    pixels = 28*28
    images_arr = []
    while True:
        try:
            img = []
            for j in range(pixels):
                pix = ord(f.read(1))
                img.append(pix / 255)
            images_arr.append(img)
        except:
            break
    f.close()
    image_sets = np.array(images_arr)
    return image_sets
def loadLabelFile(filelabel):
    f = open(filelabel, "rb")
    f.read(8)
    labels_arr = []
    while True:
        row = [0 for x in range(10)]
        try:
            label = ord(f.read(1))
            row[label] = 1
            labels_arr.append(row)
        except:
            break
    f.close()
    label_sets = np.array(labels_arr)
    return label_sets
train_images = loadImageFile("train-images.idx3-ubyte/train-images.idx3-ubyte")
train_labels = loadLabelFile("train-labels.idx1-ubyte")
test_images = loadImageFile("t10k-images.idx3-ubyte/t10k-images.idx3-ubyte")
test_labels = loadLabelFile("t10k-labels.idx1-ubyte")

x_train = train_images.reshape(train_images.shape[0], 28, 28, 1)
x_test = test_images.reshape(test_images.shape[0], 28, 28, 1)
y_train = train_labels
y_test = test_labels


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)
model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
BatchNormalization()
model.add(Dense(512))
model.add(Activation('relu'))
BatchNormalization()
model.add(Dropout(0.2))

model.add(Dense(10))

model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=100,
          epochs=10,
          verbose=2,
          validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model.weights.h5')

