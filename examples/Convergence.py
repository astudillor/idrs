#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from numpy.linalg import norm
from numpy import array, inner
import numpy as np

__all__ = ['Convergence', 'IDENTITY', 'EUCLIDEAN_NORM']


def IDENTITY(x): return x


def EUCLIDEAN_NORM(x): return norm(x)


class Convergence:
    """
    Convergence monitories the residual convergence though the callback
                function

    """

    def __init__(self, only_counter_iters=False, action=IDENTITY,
                 Norm=EUCLIDEAN_NORM, verbose=False, increment=1):
        self.iter_ = 0
        self.resVec = np.array([])
        self.action = action
        self.Norm = Norm
        self.verbose = verbose
        self.increment = increment

    def __call__(self, x):
        self.callback(x)

    def callback(self, x):
        if self.action is not None:
            rnrm = self.Norm(self.action(x))
            self.resVec = np.append(self.resVec, rnrm)
            if self.verbose:
                print("{0}\t{1}".format(self.iter_, rnrm))
        self.iter_ += self.increment

    def toFile(self, fname, header="x\ty\n"):
        try:
            with open(fname, 'w') as f:
                f.write(header)
                i = 0
                for res in self.resVec:
                    f.write("{0}\t{1}\n".format(i, res))
                    i += self.increment
        except IOError:
            print("Unable to open file {0}".format(fname))

    def reset(self, action=None, Norm=norm,
              verbose=False, increment=1):
        self.iter_ = 0
        self.resVec = []
        self.action = action
        self.Norm = Norm
        self.verbose = verbose

    def finalResidualNorm(self):
        if len(self.resVec) == 0:
            return -1
        return self.resVec[-1]

    def toArray(self):
        return array(self.resVec)

    def printInfo(self):
        print(str(self))

    def scaleResVec(self, alpha):
        self.resVec = alpha*self.resVec

    def __str__(self):
        return "Number of iter: {0}, final residual {1}".format(self.iter_, self.finalResidualNorm())

    def __len__(self):
        return len(self.resVec)

    def __getitem__(self, index):
        return self.resVec[index]
