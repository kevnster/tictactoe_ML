from .data_processing import load_file
import numpy as np

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split as tts
from sklearn.metrics import accuracy_score, confusion_matrix as cm
from sklearn.preprocessing import LabelEncoder

def fit_model(model, X, y, is_multi):
    if is_multi:
        model.fit(X, y)
    else:
        skf = StratifiedKFold(n_splits=5, random_state=69, shuffle=True)
        for i, j in skf.split(X, y):
            model.fit(X[i], y[i].ravel())
    return model

def mlp_classifier(filename, print_output=True):
    is_multi = False
    X, y = load_file(filename)
    if "multi" in filename:
        is_multi = True
    X_train, X_test, y_train, y_test = tts(X, y, random_state=69, test_size=0.30, shuffle=True)
    model = MLPClassifier(solver='adam', alpha=1e-6, max_iter=300, hidden_layer_sizes=(256,256,128,), random_state=10, activation = 'relu', early_stopping=True, validation_fraction=0.1)
    fit_model(model, X_train, y_train, is_multi)
    y_prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_prediction)
    if not is_multi:
        confusion_matrix = cm(y_test, y_prediction)
        confusion_matrix = confusion_matrix / confusion_matrix.astype(float).sum(axis=1)

    if print_output:
        print("\n--MLP Classifier--")
        print(f"File name: {filename}")
        print(f"Accuracy: {accuracy*100}%")
        if "multi" not in filename:
            print("Confusion matrix:")
            print(confusion_matrix)

    return model

def mlp_regressor(filename, print_output):
    is_multi = False
    X, y = load_file(filename)
    if "multi" in filename:
        is_multi = True
    X_train, X_test, y_train, y_test = tts(X, y, random_state=69, test_size=0.30, shuffle=True)
    model = MLPRegressor(solver='adam', alpha=1e-6, max_iter=300, hidden_layer_sizes=(256,256,128,9), random_state=10, activation = 'relu', early_stopping=True, validation_fraction=0.1)
    fit_model(model, X_train, y_train, is_multi)
    y_prediction = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    if print_output:
        print("\n--MLP Regressor--")
        print(f"File name: {filename}")
        print(f"Accuracy: {accuracy*100}%")

    return model

# def main():
#     mlp_classifier("/Users/kevspc/Documents/Code/School/CIS4930 - Intro to ML/Assignment 1/datasets-part1/tictac_single.txt")
#     mlp_classifier("/Users/kevspc/Documents/Code/School/CIS4930 - Intro to ML/Assignment 1/datasets-part1/tictac_final.txt")
#     mlp_classifier("/Users/kevspc/Documents/Code/School/CIS4930 - Intro to ML/Assignment 1/datasets-part1/tictac_multi.txt")
#     mlp_regressor("/Users/kevspc/Documents/Code/School/CIS4930 - Intro to ML/Assignment 1/datasets-part1/tictac_single.txt")
#     mlp_regressor("/Users/kevspc/Documents/Code/School/CIS4930 - Intro to ML/Assignment 1/datasets-part1/tictac_final.txt")
#     mlp_regressor("/Users/kevspc/Documents/Code/School/CIS4930 - Intro to ML/Assignment 1/datasets-part1/tictac_multi.txt")
#
# main()