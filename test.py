import math
import unittest


def toroidal_distance(L, x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx > L / 2:
        dx = L - dx
    if dy > L / 2:
        dy = L - dy

    return math.sqrt(dx ** 2 + dy ** 2)


class TestToroidalDistance(unittest.TestCase):
    def test_same_point(self):
        L = 10
        x1, y1 = 2, 3
        x2, y2 = 2, 3
        expected = 0.0
        result = toroidal_distance(L, x1, y1, x2, y2)
        self.assertAlmostEqual(result, expected)

    def test_diagonal_distance(self):
        L = 10
        x1, y1 = 2, 2
        x2, y2 = 8, 8
        expected = math.sqrt(32)
        result = toroidal_distance(L, x1, y1, x2, y2)
        self.assertAlmostEqual(result, expected)

    def test_horizontal_wrap(self):
        L = 10
        x1, y1 = 1, 5
        x2, y2 = 9, 5
        expected = 2.0
        result = toroidal_distance(L, x1, y1, x2, y2)
        self.assertAlmostEqual(result, expected)

    def test_vertical_wrap(self):
        L = 10
        x1, y1 = 5, 1
        x2, y2 = 5, 9
        expected = 2.0
        result = toroidal_distance(L, x1, y1, x2, y2)
        self.assertAlmostEqual(result, expected)

    def test_both_wrap(self):
        L = 10
        x1, y1 = 1, 1
        x2, y2 = 9, 9
        expected = math.sqrt(8)
        result = toroidal_distance(L, x1, y1, x2, y2)
        self.assertAlmostEqual(result, expected)


if __name__ == '__main__':
    unittest.main()