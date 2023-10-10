from models.linear_SVM import svm_classifer
from models.knn import knn_classifier, knn_regressor
from models.mlp import mlp_classifier, mlp_regressor
from models.linear_reg import linear_regressor

def main():
    single_filename = "datasets-part1/tictac_single.txt"
    final_filename = "datasets-part1/tictac_final.txt"
    multi_filename = "datasets-part1/tictac_multi.txt"

    svm_classifer(single_filename)
    svm_classifer(final_filename)
    svm_classifer(multi_filename)

    knn_classifier(single_filename)
    knn_classifier(final_filename)
    knn_classifier(multi_filename)

    knn_regressor(single_filename)
    knn_regressor(final_filename)
    knn_regressor(multi_filename)

    mlp_classifier(single_filename)
    mlp_classifier(final_filename)
    mlp_classifier(multi_filename)
    mlp_regressor(single_filename)
    mlp_regressor(final_filename)
    mlp_regressor(multi_filename)

    linear_regressor(single_filename)
    linear_regressor(final_filename)
    linear_regressor(multi_filename)

main()