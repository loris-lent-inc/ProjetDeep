############################################################################################################
# Cellule 1
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Paramètres
BS = 64
EPOCHS = 30
activations = ['relu', 'sigmoid', 'tanh']  # Fonctions d'activation à tester

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
activation_names = []
test_accuracies = []

# Boucle sur les différentes fonctions d'activation
for activation in activations:
    print(f"Testing activation function: {activation}")

    # Création du modèle
    model = Sequential([
        Conv2D(4, (3, 3), activation=activation, input_shape=(128, 128, 3)),
        Flatten(),
        Dense(len(train_generator.class_indices), activation="softmax")
    ])

    # Compilation du modèle
    opt = SGD(learning_rate=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Entraînement du modèle
    H = model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS, verbose=0)

    # Évaluation du modèle
    score = model.evaluate(test_generator, verbose=0)
    print(f"Test accuracy with {activation}: {score[1]}")

    # Stockage des résultats
    activation_names.append(activation)
    test_accuracies.append(score[1])

# Affichage des résultats avec matplotlib
plt.figure(figsize=(10, 6))
plt.bar(activation_names, test_accuracies, color='skyblue')
plt.xlabel("Activation Function")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs. Activation Function")
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
