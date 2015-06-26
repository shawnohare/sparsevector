import unittest
import sparsevector

class Test(unittest.TestCase):


    def test_id(self):
        d=sparsevector.SparseVector(x=1, y=2)
        e = d.__class__()  # e commonly represents the identity in a monoid
        self.assertEqual(d, d + e)
        self.assertEqual(d, e + d)

    def _right_mult(self, r, d):
        """Test multiplication (with vector on the left)."""
        data = d.data()
        desired_result_data = {k: v*r for k, v in data.items()}
        desired_result = sparsevector.SparseVector(desired_result_data)
        self.assertEqual(d*r, desired_result)

    def test_sparsity(self):
        s = sparsevector.SparseVector(x=1, y=[1, 2, 3])
        r = 0
        s = r*s
        s2 = s.sparsify()
        self.assertEqual(s2, {})

    def test_iterate_mult(self):
        inputs = [
            (1.0, sparsevector.SparseVector(x=2.0, y=3.0)),
            (0, sparsevector.SparseVector(x=0, y=2))]
        for t in inputs:
            self._right_mult(*t)

if __name__ == '__main__':
    unittest.main()
