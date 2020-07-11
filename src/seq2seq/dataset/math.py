import random

import inflect
import pandas as pd


class MathDatasetGenerator:
    digits = list("0123456789")
    operators = list("+-/*")
    operator_names = ["plus", "minus", "divided by", "times"]

    def __init__(self):
        self._inflect_engine = inflect.engine()

    def _generate_number(self, max_digits=3):
        length = random.randint(1, max_digits)
        number = ""
        for _ in range(length):
            number += str(random.randint(0, 9))

        text = self._inflect_engine.number_to_words(int(number))

        return number, text

    def _generate_operator(self):
        op_idx = random.randint(0, len(self.operators) - 1)
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
            rows.append(self._generate_expression(3))

        df = pd.DataFrame(data=rows, columns=["numbers", "text"])
        df.to_csv("data/raw/math.csv")


if __name__ == "__main__":
    mdg = MathDatasetGenerator()
    mdg.generate_expressions(1000)
