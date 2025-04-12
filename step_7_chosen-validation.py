

def k_fold_cross_validation(model_class, X, y, k=5, lr=0.001, epochs=2000):
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_size = n_samples // k
    accuracies = []
    all_y_true = []
    all_y_pred = []

    for i in range(k):
        test_idx = indices[i * fold_size : (i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = model_class(lr=lr, epochs=epochs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(f"\nFold {i+1}, Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for Fold {i+1}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    average_accuracy = np.mean(accuracies)
    print(f"\nAverage Accuracy over {k} folds: {average_accuracy:.4f}")

    print("\All Folds Classification Report:")
    print(classification_report(all_y_true, all_y_pred, zero_division=0))

    return average_accuracy
