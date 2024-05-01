import numpy as np
import unittest


def calculate_least_squares(train_y, ideal_y):
    return np.sum((train_y - ideal_y) ** 2)

def calculate_max_deviation(train_y, ideal_y):
    return np.max(np.abs(train_y - ideal_y))


class TestAnalysisFunctions(unittest.TestCase):
    def test_calculate_least_squares(self):
        # Testfall mit einfachen Arrays
        train_y = np.array([1, 2, 3])
        ideal_y = np.array([1, 2, 3])
        result = calculate_least_squares(train_y, ideal_y)
        self.assertEqual(result, 0)  # Erwartetes Ergebnis: 0, da keine Abweichung

        # Testfall mit Abweichungen
        train_y = np.array([1, 2, 3])
        ideal_y = np.array([3, 2, 1])
        result = calculate_least_squares(train_y, ideal_y)
        self.assertEqual(result, 8)  # Erwartetes Ergebnis: 8 ((2^2) + (0^2) + (2^2))

    def test_calculate_max_deviation(self):
        # Testfall mit einfachen Arrays
        train_y = np.array([1, 2, 3])
        ideal_y = np.array([1, 2, 3])
        result = calculate_max_deviation(train_y, ideal_y)
        self.assertEqual(result, 0)  # Erwartetes Ergebnis: 0, da keine Abweichung

        # Testfall mit Abweichungen
        train_y = np.array([1, 4, 3])
        ideal_y = np.array([2, 2, 2])
        result = calculate_max_deviation(train_y, ideal_y)
        self.assertEqual(result, 2)  # Erwartetes Ergebnis: 2 (maximale absolute Abweichung)

if __name__ == '__main__':
    unittest.main()
