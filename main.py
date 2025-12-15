import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def load_images(folder):
    images = []
    labels = []
    label = 0

    for person in os.listdir(folder):
        person_path = os.path.join(folder, person)
        if os.path.isdir(person_path):
            for img in os.listdir(person_path):
                img_path = os.path.join(person_path, img)
                image = cv2.imread(img_path, 0)
                image = cv2.resize(image, (100, 100))
                images.append(image.flatten())
                labels.append(label)
            label += 1

    return np.array(images), np.array(labels)

X, y = load_images("dataset")

mean_face = np.mean(X, axis=0)
X_meaned = X - mean_face

cov_matrix = np.dot(X_meaned, X_meaned.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, idx]

def get_pca_features(k):
    selected_vectors = eigenvectors[:, :k]
    eigenfaces = np.dot(X_meaned.T, selected_vectors)
    features = np.dot(X_meaned, eigenfaces)
    return features

k_values = [10, 20, 30, 40, 50]
accuracy_list = []

for k in k_values:
    X_pca = get_pca_features(k)
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.4, random_state=42)

    ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    ann.fit(X_train, y_train)

    y_pred = ann.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_list.append(acc)
    print(f"k = {k}, Accuracy = {acc}")

plt.plot(k_values, accuracy_list)
plt.xlabel("k value (Number of Eigenfaces)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k value")
plt.show()
