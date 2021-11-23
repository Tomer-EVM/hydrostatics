from hydrostatics.mesh_processing import *
import numpy as np


def test_read():
    l = read_data("data/hydrostatics_data/F50HullSTB.dxf")
    assert len(l) == 1
    assert len(l[0]) == 33 * 33


def test_conversion():
    l = read_data("data/hydrostatics_data/F50HullSTB.dxf")
    uv = convert_to_uv(l[0], (33, 33))
    uv = convert_to_uv(l[0], (9, 121))
    try:
        uv = convert_to_uv(l[0], (32, 34))
        assert False, "Operation should fail"
    except Exception as e:
        print(e)


def test_mirror():
    l = read_data("data/hydrostatics_data/F50HullSTB.dxf")
    uv = mirror_uv(convert_to_uv(l[0], (33, 33)))
    assert uv.shape == (65, 33, 3)
    assert np.all(uv[0, :, :] == uv[-1, :, :])


def test_close():
    """Need to write some tests here
    Preferably try and find edge cases where the method doesn't work
    """
    pass


def test_mesh():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 3],
            [3, 6, 7],
            [0, 3, 4],
            [3, 7, 4],
            [0, 4, 5],
            [0, 5, 1],
        ],
        dtype=int,
    )
    m = Mesh(vertices, faces)

    a, c = m.volume_properties()
    assert abs(a - 1) < 1e-8
    assert abs(c[0] - 0.5) < 1e-8
    assert abs(c[0] - c[1]) < 1e-8
    assert abs(c[0] - c[2]) < 1e-8
    assert abs(m.area() - 6) < 1e-8
    m, _ = m.cut(position=np.array([0.5, 0.5, 0.5]))
    a, c = m.volume_properties(np.array([0.5, 0.5, 0.5]))
    assert abs(a - 0.5) < 1e-8
    assert abs(c[0] - 0.5) < 1e-8
    assert abs(c[0] - c[1]) < 1e-8
    assert abs(c[2] - 0.25) < 1e-8
    assert abs(m.area() - 3) < 1e-8


def test_cull_interior():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 2.0],
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 2.0],
            [-1.0, 0.0, 2.0],
        ]
    )
    faces = np.array(
        [
            [0, 4, 2],
            [0, 2, 5],
            [0, 3, 4],
            [0, 5, 3],
            [1, 4, 2],
            [1, 2, 5],
            [1, 3, 4],
            [1, 5, 3],
        ]
    )
    m = Mesh(vertices, faces)

    m, _ = m.cut(position=np.array([0.0, 0.0, 1.5]), include_interior=False)

    v, c = m.volume_properties(np.array([0.0, 0.0, 1.5]))
    assert abs(v - 0.5625) < 1e-8
    assert abs(m.area() - 3.375) < 1e-8


def test_slice():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 3],
            [3, 6, 7],
            [0, 3, 4],
            [3, 7, 4],
            [0, 4, 5],
            [0, 5, 1],
        ]
    )
    m = Mesh(vertices, faces)
    _, s = m.cut(position=np.array([0.5, 0.5, 0.5]))
    a, c = s.area_properties()
    assert abs(a - 1) < 1e-8
    assert abs(c[0] - 0.5) < 1e-8
    assert abs(c[1] - 0.5) < 1e-8
    assert abs(c[2] - 0.5) < 1e-8
    s.centre = c
    A = s.moment_area()
    assert abs(A[0]) < 1e-8
    assert abs(A[1]) < 1e-8
    I = s.moment_inertia()
    assert abs(I[0] - 1 / 12) < 1e-8
    assert abs(I[0] - I[1]) < 1e-8
    assert abs(I[2]) < 1e-8


if __name__ == "__main__":
    test_mesh()
