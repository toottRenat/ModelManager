import unittest
import os
import sys
import numpy as np

os.chdir('..')
sys.path.append(os.getcwd())

from models import DataProcessor


class TestData(unittest.TestCase):

    def setUp(self):
        self.data = DataProcessor('for_test')
        self.data_in_path = self.data.data

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
        self.assertEqual(
            self.data.get_features().all(),
            np.array(self.data_in_path[self.data_in_path.columns.values[:-1]]).all()
        )
        self.assertEqual(
            self.data.get_labels().all(),
            np.array(self.data_in_path[self.data_in_path.columns.values[-1]]).all()
        )

        for test_size in range(1, 10):
            test_size /= 10
            _, test_features, _, test_labels = self.data.get_train_test_split(test_size=test_size)
            self.assertAlmostEqual(test_features.shape[0]/self.data_in_path.shape[0], test_size, places=5)
            self.assertAlmostEqual(test_labels.shape[0]/self.data_in_path.shape[0], test_size, places=5)


if __name__ == '__main__':
    unittest.main()

