import numpy as np
import tensorflow as tf

# Load input data from a text file
def load_data(file_path):
    X = []
    y = []
    emg_values = []
    with open(file_path, 'r') as file:
        for line in file:
            millis, emg = map(int, line.strip().split(','))
            X.append(emg)  # Only adding EMG values to X
            emg_values.append(emg)
    
    # Calculate mean of EMG values
    emg_mean = sum(emg_values) / len(emg_values)
    
    # Assign labels based on mean EMG value
    y = [1 if emg < emg_mean else 0 for emg in emg_values]
    
    return X, y

# Replace 'input_data.txt' with the path to your text file
X, y = load_data('input.txt')

# Convert data to numpy arrays
X = np.array(X)
y = np.array(y)

# Reshape X to have a single channel (for grayscale images)
X = X.reshape(-1, 1, 1, len(X))

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 1), activation='relu', input_shape=(1, 1, len(X[0]))),
    tf.keras.layers.MaxPooling2D((2, 1)),
    tf.keras.layers.Conv2D(64, (3, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 1)),
    tf.keras.layers.Conv2D(64, (3, 1), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Predict hand movement for each data point
with open('input.txt', 'r') as file:
    for line in file:
        millis, emg = map(int, line.strip().split(','))
        prediction = model.predict(np.array([emg]).reshape(1, 1, 1, 1))
        if prediction > 0.5:
            print(f"{millis},{emg} hand is at rest")
        else:
            print(f"{millis},{emg} hand is contracted")
