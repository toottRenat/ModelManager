import unittest
import os
import sys

os.chdir('..')
sys.path.append(os.getcwd())
os.chdir('tests')

from models import name_correct


class TestFunctions(unittest.TestCase):

    def test_check_name_correctness(self):

        @name_correct
        def any_func(_, name):
            return name

        with self.assertRaises(ValueError):
            for item in [[], (), {}, 1, 1.0, any_func,
                         str, ValueError, name_correct, self]:
                any_func(item, item)


if __name__ == '__main__':
    unittest.main()
