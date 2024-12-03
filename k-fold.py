from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define Stratified K-Fold cross-validator
n_splits = 5  # Number of folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to collect metrics
accuracies = []
classification_reports = []
confusion_matrices = []

# Perform Stratified K-Fold Cross-Validation
fold = 1
for train_index, test_index in skf.split(X, y):
    print(f"\nProcessing Fold {fold}/{n_splits}")
    fold += 1

    # Split the data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    class_report = classification_report(y_test, y_pred, target_names=["Onsite", "Hybrid", "Remote"], output_dict=True)
    classification_reports.append(class_report)

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    confusion_matrices.append(cm)

    # Print fold-specific results
    print(f"Accuracy for Fold {fold - 1}: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Onsite", "Hybrid", "Remote"]))
    print("Confusion Matrix:")
    print(cm)

# Compute average accuracy and other metrics
average_accuracy = sum(accuracies) / n_splits
print("\nFinal Stratified K-Fold Results:")
print(f"Average Accuracy: {average_accuracy:.2f}")
