############################################################################################################
# Cellule 1
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Paramètres
BS = 16

datagenerator = ImageDataGenerator(rescale=1. / 255)

train_generator = datagenerator.flow_from_directory(
    './Flowers/Train',
    target_size=(128, 128),
    class_mode='categorical',
    batch_size=BS,
    shuffle='true',
    color_mode='rgb'
)

test_generator = datagenerator.flow_from_directory(
    './Flowers/Test',
    target_size=(128, 128),
    class_mode='categorical',
    color_mode='rgb'
)

############################################################################################################
# Cellule 2

# Initialisation des listes pour stocker les résultats
learning_rates = []
validation_accuracies = []

# Boucle pour tester différents learning rates
for i in range(10):
    lr = 1.0 * pow(10, (-i))
    print(f"Learning rate : {lr}")

    # Création du modèle
    model = Sequential([
        Conv2D(4, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        Flatten(),
        Dense(len(train_generator.class_indices), activation="softmax")
    ])

    # Compilation du modèle
    opt = SGD(learning_rate=lr)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Entraînement du modèle
    EPOCHS = 10
    H = model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS, verbose=0)

    # Évaluation du modèle
    score = model.evaluate(test_generator, verbose=0)
    print("Test accuracy:", score[1])

    # Stockage des résultats
    learning_rates.append(lr)
    validation_accuracies.append(score[1])

# Affichage des résultats avec matplotlib
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, validation_accuracies, marker='o')
plt.xscale('log')  # Échelle logarithmique pour le learning rate
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs. Learning Rate")
plt.grid(True)
plt.show()
