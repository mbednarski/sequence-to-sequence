import numpy as np
import random
import inflect
import pandas as pd
from sklearn.model_selection import train_test_split


class MathDatasetGenerator:
    digits = list("0123456789")
    operators = list("+-/*")
    operator_names = ["plus", "minus", "divided by", "times"]

    def __init__(self):
        self._inflect_engine = inflect.engine()

    def _generate_number(self, max_digits=4):
        length = np.random.randint(1, max_digits + 1)
        number = ""
        for _ in range(length):
            number += str(np.random.randint(0, 10))

        text = self._inflect_engine.number_to_words(int(number))

        return number, text

    def _generate_operator(self):
        op_idx = np.random.randint(0, len(self.operators))
        return (self.operators[op_idx], self.operator_names[op_idx])

    def _generate_expression(self, max_length):
        exp, exp_text = self._generate_number()
        exp = [exp]
        exp_text = [exp_text]

        length = random.randint(0, max_length)

        for _ in range(length):
            op, op_text = self._generate_operator()
            exp.append(op)
            exp_text.append(op_text)

            op, op_text = self._generate_number()
            exp.append(op)
            exp_text.append(op_text)

        return " ".join(exp), " ".join(exp_text)

    def generate_expressions(self, size: int):
        rows = []
        for _ in range(size):
            rows.append(self._generate_expression(5))

        df = pd.DataFrame(data=rows, columns=["numbers", "text"])
        df = df.drop_duplicates()

        df_test, df_keep = train_test_split(df, train_size=0.2, shuffle=True)
        df_train, df_dev = train_test_split(df_keep, train_size=0.8, shuffle=True)

        df_train.to_csv("data/raw/math.train")
        df_dev.to_csv("data/raw/math.val")
        df_test.to_csv("data/raw/math.test")

        df.to_csv("data/raw/math.csv")


if __name__ == "__main__":
    mdg = MathDatasetGenerator()
    mdg.generate_expressions(50_000)
