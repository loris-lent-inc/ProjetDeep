############################################################################################################
# Cellule 1
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Paramètres fixes
LEARNING_RATE = 0.01
datagenerator = ImageDataGenerator(rescale=1. / 255)

train_dir = './Flowers/Train'
test_dir = './Flowers/Test'

############################################################################################################
# Cellule 2

# Initialisation des listes pour stocker les résultats
batch_sizes = [8, 16, 32, 64]
validation_accuracies = []

# Boucle pour tester différents batch sizes
for BS in batch_sizes:
    print(f"Batch size : {BS}")

    # Génération des données
    train_generator = datagenerator.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        class_mode='categorical',
        batch_size=BS,
        shuffle=True,
        color_mode='rgb'
    )

    test_generator = datagenerator.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        class_mode='categorical',
        batch_size=BS,  # Même batch size pour le test
        shuffle=False,
        color_mode='rgb'
    )

    # Création du modèle
    model = Sequential([
        Conv2D(4, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        Flatten(),
        Dense(len(train_generator.class_indices), activation="softmax")
    ])

    # Compilation du modèle
    opt = SGD(learning_rate=LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Entraînement du modèle
    EPOCHS = 10
    H = model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS, verbose=0)

    # Évaluation du modèle
    score = model.evaluate(test_generator, verbose=0)
    print("Test accuracy:", score[1])

    # Stockage des résultats
    validation_accuracies.append(score[1])

# Affichage des résultats avec matplotlib
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, validation_accuracies, marker='o')
plt.xlabel("Batch Size")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs. Batch Size")
plt.grid(True)
plt.show()
