import unittest
import os
import sys
import pandas as pd

os.chdir('..')
sys.path.append(os.getcwd())
os.chdir('tests')

from models import name_correct, fit_label_encoder


class Test(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
