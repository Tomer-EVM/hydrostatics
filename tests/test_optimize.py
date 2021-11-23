from hydrostatics.models import BuoyancyModel
from hydrostatics.mesh_processing import mirror_uv, close_ends
from hydrostatics.optimize import iterative_multidimensional, iterative, iterative_force
import numpy as np
import pytest


def test_multidim():
    b = BuoyancyModel()
    b.set_weight_force("hull", np.array([0, 0, 0]), 1000)
    b.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b.meshes["F50HullSTB.dxf_0"] = close_ends(mirror_uv(b.meshes["F50HullSTB.dxf_0"]))
    b.local_position["F50HullSTB.dxf_0"] = np.array([0, 0, 0])
    b.calculate_transformation()
    b.waterplane_origin = np.array([0.0, 0.0, 0.0])

    n = iterative_multidimensional(b)
    assert n <= 60
    b.calculate_results()
    assert abs(b.results.force_earth[2]) < 1e-2
    assert abs(b.results.moment_earth[0]) < 1
    assert abs(b.results.moment_earth[1]) < 1
    print(b.results.force_earth)
    print(b.results.moment_earth)


def test_iter():
    b = BuoyancyModel()
    b.set_weight_force("hull", np.array([0, 0, 0]), 1000)
    b.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b.meshes["F50HullSTB.dxf_0"] = close_ends(mirror_uv(b.meshes["F50HullSTB.dxf_0"]))
    b.local_position["F50HullSTB.dxf_0"] = np.array([0, 0, 0])
    b.calculate_transformation()
    b.waterplane_origin = np.array([0, 0, 0])

    n = iterative(b, max_iter=1000)
    assert n <= 300
    b.calculate_results()
    assert abs(b.results.force_earth[2]) < 1e-2
    assert abs(b.results.moment_earth[0]) < 1
    assert abs(b.results.moment_earth[1]) < 1
    print(b.results.force_earth)
    print(b.results.moment_earth)


@pytest.mark.skip(reason="Not an important function, may randomly fail")
def test_force():
    b = BuoyancyModel()
    b.set_weight_force("hull", np.array([0, 0, 0]), 1000)
    b.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b.meshes["F50HullSTB.dxf_0"] = close_ends(mirror_uv(b.meshes["F50HullSTB.dxf_0"]))
    b.local_position["F50HullSTB.dxf_0"] = np.array([0, 0, 0])
    b.calculate_transformation()
    b.waterplane_origin = np.array([0.0, 0.0, 0.0])
    b.calculate_results()

    n = iterative_force(b, max_iter=5000, max_time=10000000000000, damping=1)
    assert n <= 5000
    b.calculate_results()
    assert abs(b.results.force_earth[2]) < 20
    assert abs(b.results.moment_earth[0]) < 10000
    assert abs(b.results.moment_earth[1]) < 10000
    print(b.results.force_earth)
    print(b.results.moment_earth)


def test_iter_constrained():
    b = BuoyancyModel()
    b.set_weight_force("hull", np.array([5882.4, 590, 2247.4]), 2979 * 9.81)
    b.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b.meshes["F50HullSTB.dxf_0"] = close_ends(
        mirror_uv(b.meshes["F50HullSTB.dxf_0"])
    )  # Without closing faces, produces correct area
    b.load_mesh("data/hydrostatics_data/F50HullPRT.dxf")
    b.meshes["F50HullPRT.dxf_0"] = close_ends(mirror_uv(b.meshes["F50HullPRT.dxf_0"]))
    b.calculate_transformation()
    b.calculate_results()

    b.heel = 104

    iterative(b, selected=(False, True, True), max_iter=1000)

    assert b.heel == 104
    assert abs(b.results.force_earth[2]) <= 1e-2
    assert abs(b.results.moment_earth[1]) <= 1e-2


def test_capsize():
    b = BuoyancyModel()
    # b.set_weight_force("hull", np.array([5882.4,590,2247.4]), 2979*9.81)
    b.set_weight_force("approx", np.array([6000.0, 0.0, 2000.0]), 30000)
    b.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b.meshes["F50HullSTB.dxf_0"] = close_ends(
        mirror_uv(b.meshes["F50HullSTB.dxf_0"])
    )  # Without closing faces, produces correct area
    b.load_mesh("data/hydrostatics_data/F50HullPRT.dxf")
    b.meshes["F50HullPRT.dxf_0"] = close_ends(mirror_uv(b.meshes["F50HullPRT.dxf_0"]))
    b.load_mesh("data/hydrostatics_data/F50WingFlap.dxf")
    b.meshes["F50WingFlap.dxf_0"] = close_ends(mirror_uv(b.meshes["F50WingFlap.dxf_0"]))
    b.load_mesh("data/hydrostatics_data/F50WingME.dxf")
    b.meshes["F50WingME.dxf_0"] = close_ends(mirror_uv(b.meshes["F50WingME.dxf_0"]))
    b.calculate_transformation()

    b.include_interior = False

    b.heel = 104
    b.trim = 0
    b.waterplane_origin = np.array([0, -3750, 0.0])
    b.calculate_results()
    n = iterative(b, max_iter=1000)
    assert n <= 110
    assert sum(b.results.moment_earth) + sum(b.results.force_earth) <= 1
    assert 100 < b.heel < 110
    assert -1 < b.trim < 1

    b.heel = 104
    b.trim = 0
    b.waterplane_origin = np.array([0, -3750, 0.0])
    b.calculate_results()
    n = iterative_multidimensional(b, max_iter=1000)
    assert n <= 70
    assert sum(b.results.moment_earth) + sum(b.results.force_earth) <= 1
    assert 100 < b.heel < 110
    assert -1 < b.trim < 1


def test_iter_inbetween():
    b = BuoyancyModel()
    # b.set_weight_force("hull", np.array([5882.4,590,2247.4]), 2979*9.81)
    b.set_weight_force("approx", np.array([6000.0, 1000.0, 0.0]), 30000)
    b.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b.meshes["F50HullSTB.dxf_0"] = close_ends(
        mirror_uv(b.meshes["F50HullSTB.dxf_0"])
    )  # Without closing faces, produces correct area
    b.load_mesh("data/hydrostatics_data/F50HullPRT.dxf")
    b.meshes["F50HullPRT.dxf_0"] = close_ends(mirror_uv(b.meshes["F50HullPRT.dxf_0"]))
    b.calculate_transformation()

    b.heel = 90
    b.waterplane_origin = np.array(
        [
            0.0,
            0.0,
            0,
        ]
    )
    b.calculate_results()
    n = iterative(b, max_iter=1000, max_time=1000000)
    assert n < 720
    assert sum(b.results.moment_earth) + sum(b.results.force_earth) <= 1


def test_multidim_inbetween():
    b = BuoyancyModel()
    # b.set_weight_force("hull", np.array([5882.4,590,2247.4]), 2979*9.81)
    b.set_weight_force("approx", np.array([6000.0, 0.0, 2000.0]), 30000)
    b.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b.meshes["F50HullSTB.dxf_0"] = close_ends(
        mirror_uv(b.meshes["F50HullSTB.dxf_0"])
    )  # Without closing faces, produces correct area
    b.load_mesh("data/hydrostatics_data/F50HullPRT.dxf")
    b.meshes["F50HullPRT.dxf_0"] = close_ends(mirror_uv(b.meshes["F50HullPRT.dxf_0"]))
    b.calculate_transformation()

    b.heel = 90
    b.waterplane_origin = np.array(
        [
            6000.0,
            0.0,
            2000,
        ]
    )
    b.calculate_results()
    n = iterative_multidimensional(b, max_time=1000000)
    assert n < 65
    assert sum(b.results.moment_earth) + sum(b.results.force_earth) <= 1


if __name__ == "__main__":
    test_iter_inbetween()
