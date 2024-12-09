# Proposition de code pour débuter

############################################################################################################
# Cellule 1
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import SGD
BS=64
datagenerator = ImageDataGenerator(rescale=1./255)

train_generator = datagenerator.flow_from_directory('./Flowers/Train',
                                          target_size=(128, 128),
                                          class_mode='categorical',
                                          batch_size=BS,
                                          shuffle='true',
                                          color_mode='rgb')

test_generator = datagenerator.flow_from_directory('./Flowers/Test',
                                          target_size=(128, 128),
                                          class_mode='categorical',
                                          color_mode='rgb'
                                          )


############################################################################################################
# Cellule 2


model = Sequential()
model.add(Conv2D(4,(3,3),activation='tanh'))
model.add(Flatten())
model.add(Dense(len(train_generator.class_indices), activation="softmax"))
opt = SGD(learning_rate=0.01)
EPOCHS=30
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(train_generator, validation_data = test_generator, epochs=EPOCHS)
score = model.evaluate(test_generator)
print("Test accuracy : ", score)