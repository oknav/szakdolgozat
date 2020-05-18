import json
import sys


class Models:
    def __init__(self, file_path):
        self.path = file_path
        self.j_son = None
        self.models = []

    def open(self):
        try:
            with open(self.path, 'r') as file:
                self.j_son = json.load(file)
                self.read()

        except EOFError:
            print("There is no valid .json file.")
            exit()

    def read(self):
        self.models = self.j_son

    def write(self, model_num, field, value):
        self.models[model_num][field] = value

    def save(self):
        try:
            with open(self.path, 'w') as file:
                json.dump(self.models, file, indent=4)

        except EOFError:
            print("There is no valid .json file.")
            sys.exit()

    def reset(self, i):
        self.write(i, "loss", 0)
        self.write(i, "accuracy", 0)
        self.write(i, "train_time", 0)
        self.write(i, "test", {})
