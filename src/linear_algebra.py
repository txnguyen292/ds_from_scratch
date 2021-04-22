from typing import List, Tuple, Callable
import math

Vector = List[float]
Matrix = List[List[float]]


def add(v: Vector, w: Vector) -> Vector:
    """add to vectors together

    Args:
        v (Vector): input vector
        w (Vector): input vector

    Returns:
        Vector: added vector
    """
    assert len(v) == len(w), "vectors must be of same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]

def subtract(v: Vector, w: Vector) -> Vector:
    """
    Subtracts corresponding elements

    Args:
        v (Vector): input vector
        w (Vector): input vector

    Returns:
        Vector: subtracted vector
    """
    assert len(v) == len(w), "vectors must be of same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]

def dot(v: Vector, w: Vector) -> float:
    """
    Computes v_1 * w_1 + ... + v_n * w_n

    Args:
        v (Vector): input vector
        w (Vector): input vector

    Returns:
        float: dot product of input vectord
    """
    assert len(v) == len(w), "vectors must be of same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def scalar_multiply(c: float, v: Vector) -> Vector:
    """
    Multiplies every element by c

    Args:
        c (float): input scalar
        v (Vector): input vector

    Returns:
        Vector: vector ater multiplied by scalar
    """
    return [c * v_i for v_i in v]

def vector_sum(vectors: List[Vector]) -> Vector:
    """
    Sums all corresponding elements

    Args:
        vectors (List[Vector]): list of vectors to be added

    Returns:
        Vector: ?
    """
    assert vectors, "no vectors provided!"

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different size!"
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

def vector_mean(vectors: List[Vector]) -> Vector:
    """
    Computes the element-wise average

    Args:
        vectors (List[Vector]): list of vectors

    Returns:
        Vector: ?
    """
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

def sum_of_squares(v: Vector) -> float:
    """returns v_1 * v_1 + ... v_n*v_n"""
    return dot(v, v)

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or Length) of v"""
    return math.sqrt(sum_of_squares(v))


def squared_distance(v: Vector, w: Vector) -> float:
    """Comptes (v1 - w_1) * 2 + ... + (v_n - w_n) **2"""
    return sum_of_squares(subtract(v, w))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v, w))

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of cols of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A (as a Vector)"""
    return [A_i[j] for A_i in A]

def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    
    """Returns a num_rows x num_cols matrix
    whose (i, j)-th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]

def identity_matrix(n: int) -> Matrix:
    """Returns the nxn identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i==j else 0)

if __name__ == "__main__":
    assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]
    assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]
    assert dot([1, 2, 3], [4, 5, 6]) == 32
    assert sum_of_squares([1, 2, 3]) == 14
    assert magnitude([4, 3]) == 5
    assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
    assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 1]]
