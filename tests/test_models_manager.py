import unittest
import os
import sys

from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_fscore_support, \
    precision_recall_curve, average_precision_score

os.chdir('..')
sys.path.append(os.getcwd())
os.chdir('tests')

from models import DataProcessor, ModelsManager


class Test(unittest.TestCase):

    def setUp(self):
        self.data = DataProcessor('for_test')
        self.data_in_path = self.data.data

        self.mm = ModelsManager()

    def test_default_fillers(self):
        default_metric_names = [
            'average_precision_score',
            'accuracy_score',
            'roc_curve',
            'auc',
            'precision_recall_fscore_support',
            'precision_recall_curve'
        ]
        default_metrics = [
            average_precision_score,
            accuracy_score,
            roc_curve,
            auc,
            precision_recall_fscore_support,
            precision_recall_curve
        ]

        for i, metric in enumerate(default_metric_names):
            self.assertEqual(default_metrics[i], self.mm.metrics[metric])

        self.assertIn('kyoto2007', self.mm.data.keys())

        self.assertIn('lr_95-5_notime', self.mm.models.keys())
        self.assertIn('gb_95-5_notime', self.mm.models.keys())


if __name__ == '__main__':
    unittest.main()
