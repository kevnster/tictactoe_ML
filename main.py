from tic_tac_toe import TicTacToe

def main():
    game = TicTacToe()

    print("Select the model you want to play against:")
    print("1. The best ML model!")
    print("2. I want to play with another model")

    choice = int(input("Enter your choice (1/2): "))

    if choice == 1:
        game.train_model("datasets-part1/tictac_final.txt", "knn_classifier")
    elif choice == 2:
        print("Select the training dataset: ")
        print("1. single.txt")
        print("2. final.txt")
        print("3. multi.txt")
        file_choice = int(input("Enter your choice (1/2/3): "))
        files = {1: "datasets-part1/tictac_single.txt", 2: "datasets-part1/tictac_final.txt", 3: "datasets-part1/tictac_multi.txt"}
        file_name = files[file_choice]
        print("Select the model: ")
        print("1. SVM Classifier")
        print("2. MLP Classifier")
        print("3. MLP Regressor")
        print("4. KNN Classifier")
        print("5. KNN Regressor")
        model_choice = int(input("Enter your choice (1/2/3/4/5): "))
        models = {1: "svm_classifier", 2: "mlp_classifier", 3: "mlp_regressor", 4: "knn_classifier", 5: "knn_regressor"}
        model_type = models[model_choice]
        game.train_model(file_name, model_type)

    else:
        print("Invalid choice!")
        exit()

    game.play()

main()