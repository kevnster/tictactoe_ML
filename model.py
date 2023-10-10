import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.linear_model import LinearRegression

def cross_validate_classifier(X, y, classifier_type, cv=10):
    clf = train_classifier(X, y, classifier_type)
    
    # Using a custom scorer for cross_val_score based on accuracy
    scorer = make_scorer(accuracy_score)
    scores = cross_val_score(clf, X, y.ravel(), cv=cv, scoring=scorer)
    
    # Print the accuracy for each fold:
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: Accuracy: {score*100:.2f}%")
    # And the average accuracy:
    print(f"\nAverage Accuracy: {scores.mean()*100:.2f}%")

def cross_validate_regressor(X, y, regressor_type, cv=10):
    reg = train_regressor(X, y, regressor_type)
    
    # Custom scorer to evaluate the regressor as a classifier
    def custom_accuracy(y_true, y_pred):
        y_pred_rounded = np.round(y_pred)
        y_test_single = np.argmax(y_true, axis=1)
        y_pred_single = np.argmax(y_pred_rounded, axis=1)
        return accuracy_score(y_test_single, y_pred_single)
    
    scorer = make_scorer(custom_accuracy)
    scores = cross_val_score(reg, X, y, cv=cv, scoring=scorer)
    
    # Print the accuracy for each fold:
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: Accuracy: {score*100:.2f}%")
    # And the average accuracy:
    print(f"\nAverage Accuracy: {scores.mean()*100:.2f}%")

def train_classifier(X_train, y_train, classifier_type):
    if classifier_type == "linear_SVM":
        clf = SVC(kernel='linear')
    elif classifier_type == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif classifier_type == "mlp":
        clf = MLPClassifier(max_iter=1500)
    else:
        raise ValueError("Invalid classifier type.")
    
    clf.fit(X_train, y_train.ravel())
    return clf

def evaluate_classifier(clf, X_test, y_test):
    # Predicting the test set results
    y_pred = clf.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Classification Report
    cr = classification_report(y_test, y_pred, zero_division=1)
    print("\nClassification Report:\n")

def train_regressor(X_train, y_train, regressor_type):
    if regressor_type == "knn":
        reg = KNeighborsRegressor()
    elif regressor_type == "linear":
        reg = LinearRegression()
    elif regressor_type == "mlp":
        reg = MLPRegressor(max_iter=1000)
    
    reg.fit(X_train, y_train)
    return reg

def evaluate_regressor(reg, X_test, y_test):
    y_pred = reg.predict(X_test)
    # Round predictions to the nearest integer (either 0 or 1)
    y_pred_rounded = np.round(y_pred)
    
    # Convert the 9-dimensional y_test and y_pred_rounded to single labels for accuracy computation
    y_test_single = np.argmax(y_test, axis=1)
    y_pred_single = np.argmax(y_pred_rounded, axis=1)
    
    acc = accuracy_score(y_test_single, y_pred_single)
    print(f"Accuracy: {acc*100:.2f}%")