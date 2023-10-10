from .data_processing import load_file
import numpy as np

from sklearn import neighbors
from sklearn.model_selection import train_test_split as tts, cross_val_score, RepeatedKFold, StratifiedKFold 
from sklearn.metrics import confusion_matrix as cm, classification_report as cr

def fit_model(model, X, y, is_multi):
    if is_multi:
        model.fit(X, y)
    else:
        skf = StratifiedKFold(n_splits=5, random_state=69, shuffle=True)
        for i, j in skf.split(X, y):
            model.fit(X[i], y[i].ravel())
    return model

def knn_classifier(filename, print_output=True):
    number_of_classification_neighbors = 1
    is_multi = False
    X, y = load_file(filename)
    if "multi" in filename:
        is_multi = True
    X_train, X_test, y_train, y_test = tts(X, y, random_state=69, test_size=0.30, shuffle=True)

    nn = neighbors.KNeighborsClassifier(number_of_classification_neighbors, metric='euclidean')
    model = fit_model(nn, X_train, y_train, is_multi)
    y_prediction = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    if not is_multi:
        confusion_matrix = cm(y_test, y_prediction, normalize='true')

    if print_output:
        print("\n--KNN Classifier--")
        print(f"File name: {filename}")
        print(f"Number of classification neighbors: {number_of_classification_neighbors}")
        print(f"Accuracy: {accuracy*100}%")
        if "multi" not in filename:
            print("Confusion matrix:")
            print(confusion_matrix)

    return model

def knn_regressor(filename, print_output=True):
    number_of_regression_neighbors = 9
    is_multi = False
    X, y = load_file(filename)
    if "multi" in filename:
        is_multi = True
    X_train, X_test, y_train, y_test = tts(X, y, random_state=69, test_size=0.30, shuffle=True)
    knn_distance = neighbors.KNeighborsRegressor(number_of_regression_neighbors, weights='distance')
    model = fit_model(knn_distance, X_train, y_train, is_multi)
    # y_prediction = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    if print_output:
        print("\n--KNN Regressor--")
        print(f"File name: {filename}")
        print(f"Number of regression neighbors: {number_of_regression_neighbors}")
        print(f"Accuracy: {accuracy*100}%")

    return model

# def main():
#     knn_classifier("./datasets-part1/tictac_single.txt")
#     knn_classifier("./datasets-part1/tictac_final.txt")
#     knn_classifier("./datasets-part1/tictac_multi.txt")
#     knn_regressor("./datasets-part1/tictac_single.txt")
#     knn_regressor("./datasets-part1/tictac_final.txt")
#     knn_regressor("./datasets-part1/tictac_multi.txt")
#
# main()