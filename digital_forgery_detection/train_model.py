import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

# Function to load images and prepare dataset
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path)
            img = img.resize((256, 256))  # Resize to the size used in training
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(1 if 'fake' in folder else 0)  # 1 for fake, 0 for real
        except:
            continue
    return images, labels

# Load real and fake images
real_images, real_labels = load_images_from_folder('dataset/real')
fake_images, fake_labels = load_images_from_folder('dataset/fake')

# Combine images and labels
X = np.array(real_images + fake_images)
y = np.array(real_labels + fake_labels)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data (standardizing)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Save the model to a .pkl file
joblib.dump(model, 'model.pkl')