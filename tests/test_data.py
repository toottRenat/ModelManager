import unittest
import os
import sys
import pandas as pd

os.chdir('..')
sys.path.append(os.getcwd())

from models import DataProcessor


class Test(unittest.TestCase):

    def setUp(self):
        self.data = DataProcessor('for_test')
        self.data_in_path = self.data.extract_data_from_path()

    def test_checking_functions(self):
        with self.assertRaises(ValueError):
            DataProcessor(1)

            for item in [[], (), {}, 1, 1.0, '1', -1, 'qwe', True,
                         str, ValueError, self, None]:
                DataProcessor('for_test', attack_ratio=item)

            for item in [[], (), {}, 2, 2.0, '0', 0, 'qwe', True,
                         str, ValueError, self, None]:
                DataProcessor('for_test', attack_label=item)

            for item in [[], (), {}, 2, 1.0, '1', -1, 'qwe', True,
                         str, ValueError, self, None]:
                DataProcessor('for_test', processing_mode=item)

            for item in [[], (), {}, 2, 1.0, '1', -1, 'qwe', True,
                         str, ValueError, self]:
                DataProcessor('for_test', using_features=item)

            for item in [[], (), {}, 2, 1.0, -1, True,
                         str, ValueError, self]:
                DataProcessor('for_test', training_data=item)

            for item in [[], (), {}, 2, 1.0, -1, True,
                         str, ValueError, self, None]:
                DataProcessor('for_test', csv_separator=item)

            for item in [[], (), {}, 2, 1.0, '1', -1, 'qwe', True,
                         str, ValueError, self, None]:
                DataProcessor('for_test', num_classes=item)

            for item in [[], (), {}, 2, 1.0, '1', -1, 'qwe',
                         str, ValueError, self, None]:
                DataProcessor('for_test', labeled_data=item)

    def test_getters(self):
        pass


if __name__ == '__main__':
    unittest.main()

