import unittest
from factorial import factorial

class TestFactorial(unittest.TestCase):
    def test_factorial_zero(self):
        self.assertEqual(factorial(0), 1)

    def test_factorial_one(self):
        self.assertEqual(factorial(1), 1)

    def test_factorial_two(self):
        self.assertEqual(factorial(2), 2)

    def test_factorial_three(self):
        self.assertEqual(factorial(3), 6)

    def test_factorial_six(self):
        self.assertEqual(factorial(6), 720)

    def test_factorial_negative(self):
        with self.assertRaises(ValueError):
            factorial(-1)

if __name__ == '__main__':
    unittest.main()