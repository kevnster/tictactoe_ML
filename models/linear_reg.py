from .data_processing import load_file
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as tts

def linear_regressor(filename, print_output=True):
    is_multi = False
    X, y = load_file(filename)
    if "multi" in filename:
        is_multi = True

    if not is_multi:
        X_train, X_test, y_train, y_test = tts(X, y, random_state=69, test_size=0.30, shuffle=True)
        X_t = X_train.T
        X_intermediate = X_t.dot(X_train)
        X_inv = np.linalg.inv(X_intermediate)
        X_modified = X_inv.dot(X_t)
        theta = X_modified.dot(y_train)
        theta = theta.reshape(9, )

        y_prediction = X_test.dot(theta)

        y_prediction_rounded = [round(val) for val in y_prediction]
        y_test_flattened = [val for val in y_test]

        accuracy = accuracy_score(y_test_flattened, y_prediction_rounded)
    else:
        number_of_outputs = 9
        accuracy = 0

        for i in range(number_of_outputs):
            y_sub = y[:, i:i + 1]
            X_train, X_test, y_train, y_test = tts(X, y_sub, random_state=69, test_size=0.30, shuffle=True)

            X_t = X_train.T
            X_intermediate = X_t.dot(X_train)
            X_inv = np.linalg.inv(X_intermediate)
            X_modified = X_inv.dot(X_t)
            theta = X_modified.dot(y_train)
            theta = theta.reshape(9, )

            y_prediction = X_test.dot(theta)

            for i in range(len(y_prediction)):
                y_prediction[i] = round(y_prediction[i])

            trial_accuracy = accuracy_score(y_test, y_prediction)
            accuracy += trial_accuracy

        accuracy /= number_of_outputs

    if print_output:
        print("\n--Linear Regression--")
        print(f"File name: {filename}")
        print(f"Accuracy: {accuracy*100}%")

# def main():
#     linear_regressor("/Users/kevspc/Documents/Code/School/CIS4930 - Intro to ML/Assignment 1/datasets-part1/tictac_single.txt")
#     linear_regressor("/Users/kevspc/Documents/Code/School/CIS4930 - Intro to ML/Assignment 1/datasets-part1/tictac_final.txt")
#     linear_regressor("/Users/kevspc/Documents/Code/School/CIS4930 - Intro to ML/Assignment 1/datasets-part1/tictac_multi.txt")
#
# main()