from .data_processing import load_file

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix as cm

def svm_classifer(filename, print_output=True):
    is_multi = False
    X, y = load_file(filename)
    if "multi" in filename:
        is_multi = True

    if not is_multi:
        X_train, X_test, y_train, y_test = tts(X, y, random_state=69, test_size=0.30, shuffle=True)

        svm = SVC(kernel='rbf', gamma='scale', probability=True, decision_function_shape='ovr', random_state=10)
        svm.fit(X_train, y_train.ravel())

        accuracy = svm.score(X_test, y_test)

        y_prediction = svm.predict(X_test)
        confusion_matrix = cm(y_test, y_prediction)
        confusion_matrix = confusion_matrix / confusion_matrix.astype(float).sum(axis=1)
    else:
        number_of_outputs = 9
        accuracy = 0
        svm = []

        for i in range(number_of_outputs):
            y_sub = y[:, i:i + 1]
            X_train, X_test, y_train, y_test = tts(X, y_sub, random_state=69, test_size=0.30, shuffle=True)

            trial_svm = SVC(kernel='rbf', gamma='scale', probability=True, decision_function_shape='ovr', random_state=10)
            trial_svm.fit(X_train, y_train.ravel())
            svm.append(trial_svm)

            trial_accuracy = trial_svm.score(X_test, y_test)
            accuracy += trial_accuracy

        accuracy /= number_of_outputs

    if print_output:
        print("\n--SVM Classification--")
        print(f"File name: {filename}")
        print(f"Accurary: {accuracy * 100}%")
        if not is_multi:
            print("Confusion matrix:")
            print(confusion_matrix)
    
    return svm

# def main():
#     svm_classifer("/Users/kevspc/Documents/Code/School/CIS4930 - Intro to ML/Assignment 1/datasets-part1/tictac_single.txt")
#     svm_classifer("/Users/kevspc/Documents/Code/School/CIS4930 - Intro to ML/Assignment 1/datasets-part1/tictac_final.txt")
#     svm_classifer("/Users/kevspc/Documents/Code/School/CIS4930 - Intro to ML/Assignment 1/datasets-part1/tictac_multi.txt")
#
# main()