import random
import pickle

from models.linear_SVM import svm_classifer
from models.mlp import mlp_classifier, mlp_regressor
from models.knn import knn_classifier, knn_regressor

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.model = None

    def train_model(self, filename, model_type):
        models = {
            "svm_classifier": svm_classifer,
            "mlp_classifier": mlp_classifier,
            "mlp_regressor": mlp_regressor,
            "knn_classifier": knn_classifier,
            "knn_regressor": knn_regressor
        }

        if model_type in models:
            self.model = models[model_type](filename, print_output=False)

    def display_board(self):
        for i in range(0, 9, 3):
            print(self.board[i], '|', self.board[i+1], '|', self.board[i+2])

    def reset_board(self):
        self.board = [' '] * 9

    def is_winner(self, player):
        win_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for combo in win_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] == player:
                return True
        return False

    def get_empty_positions(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def make_move(self, position, player):
        if self.board[position] == ' ':
            self.board[position] = player
            return True
        return False

    def ml_move(self):
        empty_positions = self.get_empty_positions()
        return random.choice(empty_positions)

    def play(self):
        while True:
            turn = 'X'
            for _ in range(9):
                self.display_board()
                if turn == 'X':
                    position = int(input("Enter your move (0-8): "))
                else:
                    position = self.ml_move()
                    print(f"Computer chose position {position}")
                if self.make_move(position, turn):
                    if self.is_winner(turn):
                        self.display_board()
                        print(f"{turn} wins!")
                        break
                    turn = 'O' if turn == 'X' else 'X'
            else:
                self.display_board()
                print("It's a tie!")

            play_again = input("Do you want to play again? (y/n): ").lower()
            if play_again != 'y':
                break
            self.reset_board()