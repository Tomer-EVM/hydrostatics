from hydrostatics.transformations import (
    get_closest_divisors,
    transformation_matrix,
)  # , clip_points

# from hydrostatics.mesh_processing import Tri
import numpy as np


def test_get_closest():
    # Basic tests for reference
    a, b = get_closest_divisors(650)
    assert a + b == 51
    a, b = get_closest_divisors(21608)
    assert a + b == 294

    for n in range(100):
        get_closest_divisors(n * np.random.randint(2000) + 1)


def test_rotation():
    assert np.all(transformation_matrix(0, 0, 0) == np.eye(3))
    assert np.all(
        transformation_matrix(0, 0, 90) - np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        < 1e-9
    )


"""
def test_clip():
    a = np.array([1.,1.,1.])
    b = np.array([-1.,1.,1.])
    c = np.array([-1.,-1.,1.])
    t,e = clip_points(a,b,c)
    assert t == [] and e == []
    t,e = clip_points(a,b,c,np.array([1.,0.,0.]))
    assert len(t) == 2
    assert len(e) == 1
    a = np.array([1.,1.,-1.])
    b = np.array([-1.,1.,-1.])
    c = np.array([-1.,-1.,-1.])
    t,e = clip_points(a,b,c)
    assert t[0].volume() == Tri(a,b,c).volume()
"""


def test_rotation_performance():
    for _ in range(1000):
        transformation_matrix(
            np.random.rand() * 360, np.random.rand() * 360, np.random.rand() * 360
        )


"""
def test_clip_performance():
    for _ in range(1000):
        a = np.random.rand(3)*100
        b = np.random.rand(3)*100
        c = np.random.rand(3)*100
        n = np.random.rand(3)*100
        p = np.random.rand(3)*100
        clip_points(a,b,c,n,p)
"""
