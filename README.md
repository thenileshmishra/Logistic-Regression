
# Logistic Regression for Salary Classification

## **Project Overview**
This project implements Logistic Regression from scratch in Python to classify whether an individual's salary is greater than $100,000 (`High Salary`) based on features such as education, job title, gender, and location. The project avoids using libraries like `scikit-learn` for training, focusing on understanding the core concepts of Logistic Regression.

---

## **Dataset**
The dataset includes 1000 data points with the following features:
- **Education**: Categorical (e.g., High School, Bachelor, Master, PhD).
- **Job Title**: Categorical (e.g., Analyst, Engineer, Manager, Director).
- **Gender**: Binary (Male/Female).
- **Location**: Categorical (Rural, Suburban, Urban).
- **Salary**: Continuous values representing annual salaries.

### **Target Variable**
- **High_Salary**: Binary classification target.
  - `1`: Salary > $100,000.
  - `0`: Salary ≤ $100,000.

### **Preprocessing**
Categorical variables are encoded into numerical values. For example:
- Education levels: `{'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}`.
- Job titles, gender, and location are similarly encoded.

---

## **Code Implementation**
### **1. Feature and Target Separation**
The features (`X`) include all encoded columns except the target (`High_Salary`), which is stored as the target variable (`y`).

```python
X = data.drop(columns=["Education", "Gender", "Location", "Job_Title", "Salary", "High_Salary"])
y = data["High_Salary"].values
```

### **2. Train-Test Split**
The dataset is split into training and testing sets using a custom function with a default 80-20 split.

```python
def split_data(X, y, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)
    X_train = X[:split_index].values
    X_test = X[split_index:].values
    y_train = y[:split_index]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test
```

---

### **3. Logistic Regression Implementation**
A custom `LogisticRegression` class is implemented with the following methods:
1. **`fit(X, y)`**: Trains the model using gradient descent and Binary Cross-Entropy loss.
2. **`predict_prob(X)`**: Predicts probabilities for the target class (`1`).
3. **`predict(X, threshold=0.5)`**: Classifies data points using a probability threshold (default: `0.5`).
4. **`evaluate(X, y)`**: Calculates accuracy to evaluate the model.

#### **Key Features**
- Sigmoid activation function to predict probabilities.
- Binary Cross-Entropy loss for optimization.
- Evaluation using accuracy.

```python
class LogisticRegression:
    def fit(self, X, y):
        # Train the model using gradient descent
        pass

    def predict_prob(self, X):
        # Predict probabilities
        pass

    def predict(self, X, threshold=0.5):
        # Classify based on a probability threshold
        pass

    def evaluate(self, X, y):
        # Calculate accuracy
        pass
```

---

### **4. Model Training and Evaluation**
The model is trained on the training set and evaluated on the testing set for accuracy.

```python
# Train the model
model = LogisticRegression(learning_rate=0.01, epoch=1000)
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")
```

Sample predictions are displayed for review.

---

## **How to Run the Code**
1. Ensure the dataset (`salary_classification_data.csv`) is present in the working directory.
2. Run the script using Python:
   ```bash
   python main.py
   ```
3. Review the output for accuracy and sample predictions.

---

## **Project Highlights**
- **Custom Logistic Regression**: Built from scratch to predict binary outcomes.
- **Data Preprocessing**: Encodes categorical variables and includes a binary classification target.
- **Evaluation Metrics**: Accuracy as the performance metric.
