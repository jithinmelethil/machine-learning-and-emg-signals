from sklearn.neighbors import KNeighborsClassifier

# Load input data from a text file
def load_data(file_path):
    X = []
    y = []
    with open(file_path, 'r') as file:
        for line in file:
            millis, emg = map(int, line.strip().split(','))
            X.append([millis, emg])
            # Assuming the label is 1 for movement and 0 for no movement
            y.append(1 if emg < 200 else 0)
    return X, y

# Replace 'input_data.txt' with the path to your text file
X, y = load_data('input.txt')

# Initialize and train the classifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X, y)

# Predict hand movement for each data point
with open('input.txt', 'r') as file:
    for line in file:
        millis, emg = map(int, line.strip().split(','))
        prediction = clf.predict([[millis, emg]])
        if prediction == 1:
            print(f"{millis},{emg} hand is at rest")
        else:
            print(f"{millis},{emg} hand is contracted")
