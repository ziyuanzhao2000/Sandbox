from unittest import TestCase
from data_utils import column_normalize
import pandas as pd
import numpy as np

class TestColumnNormalize(TestCase):
    def test_column_normalize_empty(self):
        df = pd.DataFrame([])
        df2 = df.copy()
        column_normalize(df2, "foo")
        self.assertTrue(df.equals(df2))

    def test_column_normalize_column_not_found(self):
        df = pd.DataFrame(np.random.rand(3,3))
        df2 = df.copy()
        column_normalize(df2, 3)
        self.assertTrue(df.equals(df2))

    def test_column_normalize_column(self):
        df = pd.DataFrame([[1,2,3],[1,2,3],[2,4,6]])
        column_normalize(df, 2)
        self.assertTrue(df.equals(pd.DataFrame([[1,2,0.0],[1,2,0.0],[2,4,1.0]])))