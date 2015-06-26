## Introduction

The sparsevector.py module includes the SparseVector class, which
implements basic algebraic operations on sparse representations of
vector objects. The class assumes **Python 3.3** or greater as it makes use of the `yield from` syntax.  

### Sparseness

A prototypical example of a vector is an element in the n-fold
set product of the real numbers R, i.e., an n-tuple of real numbers.
A sparse vector is a data structure that only records non-zero components
of a vector. 

One simple way to represent a sparse vector is as an
associative array or key-value pairs where the key is
a component index and the value is a field element.
For example, the vector (0, 1, 0) has a sparse representation
{2: 1}.

### Sparse module elements

Despite its name, a SparseVector object can be any
elements from a finite product of R-modules for some ring R.
If M = M1 x M2 x ... x Mn for R-modules Mi, then a typical
element might look something like {Mi1: mi1, ..., Mil: mil}.
