"""Provides basic sparse vector operations over Python dictionaries."""
import functools
import operator
from copy import deepcopy
import logging
import math


class SparseVector(object):
    """Equips a dictionary with + and * operators."""
    def __init__(self, input_dict={}, **kwargs):
        """Input d is a dictionary."""
        d = deepcopy(input_dict)  # recursively copies
        d.update(kwargs)
        # sparsify underlying data
        self._data = {k: v for k, v in d.items() if v}

    def keys(self):
        return set(self._data.keys())

    def values(self):
        return self._data.values()

    def __iter__(self):
        yield from self._data

    def __getitem__(self, key):
        return self._data.get(key, 0)

    def __add__(self, x):
        d1 = self.data()
        d2 = x.data()
        tmp_d = {}
        # get set union of keys from d and x
        all_keys = self.keys() | (x.keys())
        for key in all_keys:
            v_self = d1.get(key)
            v_x = d2.get(key)
            if v_self is None:
                tmp_d[key] = v_x
            elif v_x is None:
                tmp_d[key] = v_self
            else:
                tmp_d[key] = v_self + v_x
        return self.__class__(tmp_d)

    def dot(self, other):
        """
        Compute the standard Euclidean inner product of two sparse vectors.
        """
        return sum((self[k]*other[k] for k in self))

    def norm(self):
        """
        Compute the norm of self induced from the inner product defined
        in the dot function.
        """
        return math.sqrt(self.dot(self))

    def distance(self, other):
        return self.norm(self - other)

    def __sub__(self, other):
        return self + (-1)*other

    def __len__(self):
        return len(self._data)

    def __mul__(self, r):
        return self.__class__({k: v*r for k, v in self._data.items()})

    def __rmul__(self, r):
        return self.__class__({k: r*v for k, v in self._data.items()})

    def __repr__(self):
        return self._data.__repr__()

    def __eq__(self, other):
        """Determines whether two SparseVector objects are equal."""
        if isinstance(other, type(self)):
            result = self.data() == other.data()
        elif isinstance(other, type(self.data())):
            result = self.data() == other
        else:
            result = NotImplemented
        return result

    def __ne__(self, other):
        eq_result = self.__eq__(other)
        if eq_result is NotImplemented:
            result = eq_result
        else:
            result = not eq_result
        return result

    def data(self):
        return self._data

    def sparsify(self):
        """
        Return a copy of self where (key, value) pairs with empty values
        are removed.
        """
        s = {k: v for k, v in self.data().items() if v}
        return self.__class__(s)


def summation(itr):
    """Generalized sum over a collection of monoid elements."""
    return functools.reduce(operator.add, itr, SparseVector())


class Test(object):
    def run(self):
        self._iterate_id_test()
        self._iterate_mult_test()
        self._sparsity_test()

    def _id_test(self, d=SparseVector(x=1, y=2)):
        e = self.__class__()  # e commonly represents the identity in a monoid
        passed = d == d + e
        print('Passed {}+{}...{}'.format(d, e, passed))

    def _mult_test(self, r, d):
        """Test multiplication (with vector on the left)."""
        data = d.data()
        desired_result = {k: v*r for k, v in data.items()}
        passed = d * r == desired_result
        print('Passed {}*{}...{}'.format(r, d, passed))

    def _sparsity_test(self):
        s = SparseVector(x=1, y=[1, 2, 3])
        r = 0
        s = r*s
        s2 = s.sparsify()
        passed = s2 == {}
        print("Passed sparsify test for {}...{}".format(s, passed))

    def _iterate_id_test(self):
        inputs = [
            SparseVector(x=1, y=2),
            SparseVector(x=[1, 2], y={}),
            SparseVector()]
        for input in inputs:
            self._id_test(input)

    def _iterate_mult_test(self):
        inputs = [
            (1.0, SparseVector(x=2.0, y=3.0)),
            (0, SparseVector(x=0, y=2))]
        for t in inputs:
            self._mult_test(*t)


def main():
    test = Test()
    test.run()

if __name__ == '__main__':
    main()
