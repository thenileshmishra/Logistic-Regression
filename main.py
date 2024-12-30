import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("salary_classification_data.csv")

# Encoding categorical variables
education_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
data['Education_Encoding'] = data["Education"].map(education_mapping)

Job_title_mapping = {"Analyst": 1, "Engineer": 2, "Manager": 3, "Director": 4}
data["Job_title_Encoding"] = data["Job_Title"].map(Job_title_mapping)

Gender_mapping = {"Male": 1, "Female": 0}
data["Sex"] = data["Gender"].map(Gender_mapping)

loc_mapping = {"Rural": 1, "Suburban": 2, "Urban": 3}
data["locations"] = data["Location"].map(loc_mapping)

# Binary classification (e.g., High Salary > 100,000 or not)
threshold = 100000
data["High_Salary"] = (data["Salary"] > threshold).astype(int)

# Features and target
X = data.drop(columns=["Education", "Gender", "Location", "Job_Title", "Salary", "High_Salary"])
y = data["High_Salary"].values

# Train-test split function
def split_data(X, y, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)
    X_train = X[:split_index].values
    X_test = X[split_index:].values
    y_train = y[:split_index]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test

# Logistic Regression implementation
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epoch):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_prob(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_prob(X)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

# Split the data
X_train, X_test, y_train, y_test = split_data(X, y, split_ratio=0.8)

# Train the model
model = LogisticRegression(learning_rate=0.01, epoch=1000)
model.fit(X_train, y_train)

# Predict and evaluate
accuracy = model.evaluate(X_test, y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Display sample predictions
y_pred = model.predict(X_test[:5])
for actual, predicted in zip(y_test[:5], y_pred):
    print(f"Actual: {actual}, Predicted: {predicted}")