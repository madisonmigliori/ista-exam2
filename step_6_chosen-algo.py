import numpy as np 

class MultinomialLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None

    # One hot encoding the measures
    def one_hot(self, y, num_classes): 
        return np.eye(num_classes)[y]

    def softmax(self, z):
        z -= np.max(z, axis=1, keepdims=True)
        exp_scores = np.exp(z)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def cross_entropy(self, probs, Y):
        return -np.mean(np.sum(Y * np.log(probs + 1e-9), axis=1))

    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        self.X_train = X
        num_samples, num_features = X.shape
        self.num_classes = len(np.unique(y))

        Y = self.one_hot(y, self.num_classes)

        #
        self.weights = np.random.randn(num_features, self.num_classes)

        for epoch in range(self.epochs):
            logits = X @ self.weights  # shape: (N, K)
            probs = self.softmax(logits)  # shape: (N, K)
            loss = self.cross_entropy(probs, Y)

            grad = X.T @ (probs - Y) / num_samples
            self.weights -= self.lr * grad

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict_proba(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        logits = X @ self.weights
        return self.softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# One hot coding the Measure Names over the Predicted Rate & Number of Discharges
X = hospital_df_clean[["Predicted Readmission Rate", "Number of Discharges"]].values
y_raw = hospital_df_clean["Measure Name"].astype("category")
y = y_raw.cat.codes.values 

model = MultinomialLogisticRegression(lr=0.01, epochs=1000)
model.fit(X, y)

preds = model.predict(X)

accuracy = np.mean(preds == y)
print("Accuracy:", accuracy)