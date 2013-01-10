"""
Comparison operators for pairwise image tasks (e.g. lfw, pubfig).
"""

import numpy as np


class Comparison(object):
    def get_num_features(self,shp):
        return shp[1] * shp[2] * shp[3]


class Concatenate(Comparison):
    def get_num_features(self, shp):
        return 2 * shp[1] * shp[2] * shp[3]
    def __call__(self, x, y):
        return np.concatenate([x.flatten(), y.flatten()])
concatenate = Concatenate()


class Mult(Comparison):
    def __call__(self, x, y):
        return x.flatten() * y.flatten()
mult = Mult()


class Diff(Comparison):
    def __call__(self, x, y):
        return x.flatten() - y.flatten()
diff = Diff()


class Absdiff(Comparison):
    def __call__(self, x, y):
        return np.abs(x.flatten() - y.flatten())
absdiff = Absdiff()


class Sqrtabsdiff(Comparison):
    def __call__(self, x, y):
        return np.sqrt(np.abs(x.flatten() - y.flatten()))
sqrtabsdiff = Sqrtabsdiff()


class Sqdiff(Comparison):
    def __call__(self, x, y):
        return (x.flatten() - y.flatten())**2
sqdiff = Sqdiff()
