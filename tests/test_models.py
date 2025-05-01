from hydrostatics.models import BuoyancyModel
from hydrostatics.mesh_processing import mirror_uv, close_ends
import numpy as np

try:
    import pyvista as pv
except:
    pv = None


def test_add_weight():
    b = BuoyancyModel()
    b.set_weight_force("weight", np.array([1, 1, 1]), 1)


def test_load_mesh():
    b = BuoyancyModel()
    b.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b.meshes["F50HullSTB.dxf_0"] = close_ends(mirror_uv(b.meshes["F50HullSTB.dxf_0"]))


def test_transformation():
    b = BuoyancyModel()
    b.calculate_transformation()


def test_cut():
    b = BuoyancyModel()
    b.cut()


def test_full_calculation():
    b = BuoyancyModel()
    b.calculate_results()


def test_cube():
    h = BuoyancyModel()
    h.meshes["cube"] = close_ends(
        mirror_uv(
            np.flip(
                np.array(
                    [
                        [[1000, 0, 1000], [-1000, 0, 1000], [-1000, 0, 1000]],
                        [[1000, 1000, 1000], [-1000, 1000, 1000], [-1000, 0, 1000]],
                        [[1000, 1000, -1000], [-1000, 1000, -1000], [-1000, 0, -1000]],
                        [[1000, 0, -1000], [-1000, 0, -1000], [-1000, 0, -1000]],
                    ],
                    dtype="float64",
                ),
                0,
            )
        )
    )
    h.active_mesh["cube"] = True
    h.show_mesh["cube"] = True
    h.local_position["cube"] = np.array([0.0, 0.0, 0.0])
    h.local_rotation["cube"] = np.array([0.0, 0.0, 0.0])
    h.calculate_transformation()
    h.calculate_results()
    if pv:
        p = pv.Plotter()
        for mesh in h.plot_below_surface():
            p.add_mesh(mesh, culling="back")

    assert h.results.current
    assert not h.results.partial
    assert np.all(h.results.force == h.results.buoyancy_force)
    assert np.all(h.results.moment == h.results.buoyancy_moment)
    assert np.all(h.results.centre_of_gravity == np.array([0.0, 0.0, 0.0]))
    assert abs(h.results.volume - 2000 * 2000 * 1000) < 1e-4
    assert np.all(abs(h.results.volume_centroid - np.array([0, 0, -500])) < 1e-4)
    assert abs(h.results.wetted_surface_area - 12000000) < 1e-4
    assert abs(h.results.waterplane_area - 2000 * 2000) < 1e-4
    assert h.results.waterplane_area > 1e-9
    assert np.all(abs(h.results.centre_of_flotation) < 1e-4)
    assert abs(h.results.Ixx - 1333333333333.3333) < 1e-3
    assert abs(h.results.Iyy - h.results.Ixx) < 1e-3
    assert abs(h.results.Iyy - h.results.Iu) < 1e-4
    assert abs(h.results.Iyy - h.results.Iv) < 1e-4
    # assert abs(h.results.theta_principle) < 1 # When Iyy and Ixx are very close, and Ixy is very small, we get massive error in the angle. This is likely unavoidable
    assert abs(h.results.Ixy) < 1e-4
    assert h.results.state == "Floating"
    assert abs(h.results.bounds[2, 0] + 1000) < 1e-4
    assert abs(h.results.Lwl - 2000) < 1e-4
    assert abs(h.results.Bwl - 2000) < 1e-4
    assert abs(h.results.Cwp - 1) < 1e-4


def test_against_excel():
    b = BuoyancyModel()
    b.set_weight_force("hull", np.array([5882.4, 590, 2247.4]), 2979 * 9.81)
    b.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b.meshes["F50HullSTB.dxf_0"] = mirror_uv(
        b.meshes["F50HullSTB.dxf_0"]
    )  # Without closing faces, produces correct area
    b.local_position["F50HullSTB.dxf_0"] = np.array([0, 0, 0])
    b.local_rotation["F50HullSTB.dxf_0"] = np.array([0, 0, 0])
    b.calculate_transformation()
    b.calculate_results()

    assert abs((b.results.volume - 1195654202.10127) / b.results.volume) < 1e-7
    # assert abs((b.results.bounds[2,0] + 265.438659667968)/b.results.bounds[2,0]) < 1e-8
    assert abs((b.results.Lwl - 13659.6181685288) / b.results.Lwl) < 1e-7
    assert abs((b.results.Bwl - 690.086370260558) / b.results.Bwl) < 1e-7
    assert (
        abs(
            (b.results.wetted_surface_area - 10136608.4581886)
            / b.results.wetted_surface_area
        )
        < 1e-8
    )
    assert (
        abs((b.results.waterplane_area - 7752889.78609542) / b.results.waterplane_area)
        < 1e-8
    )
    assert abs((b.results.Cwp - 0.822472906915916) / b.results.Cwp) < 1e-7
    assert np.all(
        np.isclose(
            b.results.volume_centroid,
            np.array([6193.01396622061, -3737.52760421445, -93.6758117708258]),
        )
    )
    b.meshes["F50HullSTB.dxf_0"] = close_ends(
        mirror_uv(b.meshes["F50HullSTB.dxf_0"])
    )  # Without closing faces, produces correct area
    b.calculate_transformation()
    b.calculate_results()
    assert abs((b.results.BMt - 0.197019473752493 * 1000) / b.results.BMt) < 1e-7
    assert abs((b.results.BMl - 81.7520382810099 * 1000) / b.results.BMl) < 1e-7
    assert abs((b.results.GMt + 2.14409903627079 * 1000) / b.results.GMt) < 1e-4
    assert abs((b.results.GMl - 79.4109197709866 * 1000) / b.results.GMl) < 1e-6
    assert np.all(
        np.isclose(
            b.results.centre_of_flotation[0:2],
            np.array([6564.94484608725, -3737.46471134494]),
        )
    )
    assert abs((b.results.Ixx - 235567161687.948) / b.results.Ixx) < 1e-8
    assert abs((b.results.Iyy - 97747168101033) / b.results.Iyy) < 1e-7
    assert abs((b.results.Ixy - 475400280.56102) / b.results.Ixy) < 1e-4
    assert abs((b.results.Iu - 97747168103350.7) / b.results.Iu) < 1e-7
    assert abs((b.results.Iv - 235567159370.216) / b.results.Iv) < 1e-7
    assert abs((b.results.Ixx_origin - 108532912681718) / b.results.Ixx_origin) < 1e-8
    assert abs((b.results.Iyy_origin - 431885094998770) / b.results.Iyy_origin) < 1e-7
    assert abs((b.results.Ixy_origin + 190226364242791) / b.results.Ixy_origin) < 1e-7
    assert abs((b.results.Ax_origin + 28976151986.4783) / b.results.Ax_origin) < 1e-7
    assert abs((b.results.Ay_origin - 50897293843.5096) / b.results.Ay_origin) < 1e-7
    assert (
        abs((b.results.theta_principle - 45.0002793352728) / b.results.theta_principle)
        < 1e-4
    )


def test_run_performance():
    b = BuoyancyModel()
    b.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b.meshes["F50HullSTB.dxf_0"] = close_ends(mirror_uv(b.meshes["F50HullSTB.dxf_0"]))
    b.calculate_transformation()
    for _ in range(10):
        b.heel = np.random.rand() * 100
        b.trim = np.random.rand() * 100
        b.height = np.random.rand() * 100
        b.run()


def test_run_vpp():
    b = BuoyancyModel()
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
    for _ in range(10):
        np.random.seed(0)
        for _ in range(10):
            x = (0, 0, 0, 0)
            # b.set_waterplane_vpp()
            b.run_vpp(x[0], x[1], x[2], x[3])


def test_multiple_angles_excel():
    b = BuoyancyModel()
    b.set_weight_force("hull", np.array([5882.4, 590, 2247.4]), 2979 * 9.81)
    b.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b.meshes["F50HullSTB.dxf_0"] = mirror_uv(
        b.meshes["F50HullSTB.dxf_0"]
    )  # Without closing faces, produces correct area
    b.load_mesh("data/hydrostatics_data/F50HullPRT.dxf")
    b.meshes["F50HullPRT.dxf_0"] = mirror_uv(b.meshes["F50HullPRT.dxf_0"])
    b.calculate_transformation()
    b.calculate_results()

    b2 = BuoyancyModel()
    b2.set_weight_force("hull", np.array([5882.4, 590, 2247.4]), 2979 * 9.81)
    b2.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b2.meshes["F50HullSTB.dxf_0"] = close_ends(
        mirror_uv(b.meshes["F50HullSTB.dxf_0"])
    )  # Without closing faces, produces correct area
    b2.load_mesh("data/hydrostatics_data/F50HullPRT.dxf")
    b2.meshes["F50HullPRT.dxf_0"] = close_ends(mirror_uv(b.meshes["F50HullPRT.dxf_0"]))
    b2.calculate_transformation()
    b2.calculate_results()

    b.set_waterplane(
        0, -0.288516761, 0, -35.4571348758036, cor=np.array([5872.8, -3737.5, 0.0])
    )
    b.calculate_results()
    b2.set_waterplane(
        0, -0.288516761, 0, -35.4571348758036, cor=np.array([5872.8, -3737.5, 0.0])
    )
    b2.calculate_results()
    assert abs((b2.results.volume - 2896637471.79749) / b2.results.volume) < 0.1
    assert (
        abs(
            (b.results.wetted_surface_area - 21976694.4968163)
            / b.results.wetted_surface_area
        )
        < 1e-2
    )
    assert (
        abs((b2.results.waterplane_area - 15882799.0013372) / b.results.waterplane_area)
        < 1e-2
    )
    assert (
        abs(
            (b2.results.volume_centroid[0] - 5905.98461176854)
            / b2.results.volume_centroid[0]
        )
        < 1e-2
    )
    # assert abs((b.results.volume_centroid[1] + 0.0740301958530801)/b.results.volume_centroid[1]) < 1e-2
    assert (
        abs(
            (b2.results.volume_centroid[2] + 73.4679357849621)
            / b2.results.volume_centroid[2]
        )
        < 1e-2
    )
    # assert abs((b.results.centre_of_flotation[0] - 6488.6388814703)/b.results.centre_of_flotation[0]) < 1e-2
    # assert abs((b.results.centre_of_flotation[1] - 0.00571871237405039)/b.results.centre_of_flotation[1]) < 1e-2
    assert abs((b2.results.BMt - 76.76876251541 * 1000) / b.results.BMt) < 1e-2
    assert abs((b2.results.GMt - 74.4742664205135 * 1000) / b.results.GMt) < 1e-2
    assert abs((b2.results.Ixx - 222371274165659) / b.results.Ixx) < 1e-2
    assert abs((b2.results.Iyy - 201113327696802) / b.results.Iyy) < 1e-2
    """
    b.set_waterplane(5, -0.855433863325347, 0, -212.198135649001, cor=np.array([5872.8, -3737.5, 0.0]))
    b.calculate_results()
    b2.set_waterplane(5, -0.855433863325347, 0, -212.198135649001, cor=np.array([5872.8, -3737.5, 0.0]))
    b2.calculate_results()
    abs((b2.results.volume - 2895920451.29837)/b.results.volume) < 1e-2
    abs((b.results.wetted_surface_area - 15972284.3202597)/b.results.wetted_surface_area) < 1e-2
    abs((b2.results.waterplane_area - 8999972.43593346)/b.results.waterplane_area) < 1e-2
    abs((b2.results.volume_centroid[0]-5873.96508003822)/b.results.volume_centroid[0]) < 1e-2
    abs((b2.results.volume_centroid[1] + 3749.63110360318)/b.results.volume_centroid[1]) < 1e-2
    abs((b2.results.volume_centroid[2] - 25.9710247844337)/b.results.volume_centroid[2]) < 1e-2
    abs((b.results.centre_of_flotation[0] - 6731.50103239232)/b.results.centre_of_flotation[0]) < 1e-2
    abs((b.results.centre_of_flotation[1] + 3758.12381338564)/b.results.centre_of_flotation[1]) < 1e-2
    abs((b.results.BMt - 0.113125494940765)/b.results.BMt) < 1e-2
    abs((b.results.GMt + 2.45099420636934)/b.results.GMt) < 1e-2
    abs((b.results.Ixx - 327602434362.211)/b.results.Ixx) < 1e-2
    abs((b.results.Iyy - 129462221143590)/b.results.Iyy) < 1e-2

    b.set_waterplane(10, -0.878415244627889, 0, -210.130539995327, cor=np.array([5872.8, -3737.5, 0.0]))
    b.calculate_results()
    b2.set_waterplane(10, -0.878415244627889, 0, -210.130539995327, cor=np.array([5872.8, -3737.5, 0.0]))
    b2.calculate_results()
    abs((b.results.volume - 2897548297.27542)/b.results.volume) < 1e-2
    abs((b.results.wetted_surface_area - 15974304.0839776)/b.results.wetted_surface_area) < 1e-2
    abs((b.results.waterplane_area - 9087500.18461355)/b.results.waterplane_area) < 1e-2
    abs((b.results.volume_centroid[0]-5849.01489176109)/b.results.volume_centroid[0]) < 1e-2
    abs((b.results.volume_centroid[1] + 3761.88266459805)/b.results.volume_centroid[1]) < 1e-2
    abs((b.results.volume_centroid[2] - 24.6022898642009)/b.results.volume_centroid[2]) < 1e-2
    abs((b.results.centre_of_flotation[0] - 6717.68490489396)/b.results.centre_of_flotation[0]) < 1e-2
    abs((b.results.centre_of_flotation[1] + 3778.93232837365)/b.results.centre_of_flotation[1]) < 1e-2
    abs((b.results.BMt - 0.116959998593337)/b.results.BMt) < 1e-2
    abs((b.results.GMt + 2.79767728864839)/b.results.GMt) < 1e-2
    abs((b.results.Ixx - 338897244773.458)/b.results.Ixx) < 1e-2
    abs((b.results.Iyy - 130389386888325)/b.results.Iyy) < 1e-2
    
    b.set_waterplane(40, -0.727982909099343, -151.565121781228, cor=np.array([5872.8, -3737.5, 0.0]))
    b.calculate_results()
    assert abs((b.results.volume - 2906122622.31428)/b.results.volume) < 1e-2
    assert abs((b.results.wetted_surface_area - 17352704.3346527)/b.results.wetted_surface_area) < 1e-2
    assert abs((b.results.waterplane_area - )/b.results.waterplane_area) < 1e-2
    assert abs((b.results.volume_centroid[0]-)/b.results.volume_centroid[0]) < 1e-2
    assert abs((b.results.volume_centroid[1] - )/b.results.volume_centroid[1]) < 1e-2
    assert abs((b.results.volume_centroid[2] - )/b.results.volume_centroid[2]) < 1e-2
    assert abs((b.results.centre_of_flotation[0] - )/b.results.centre_of_flotation[0]) < 1e-2
    assert abs((b.results.centre_of_flotation[1] - )/b.results.centre_of_flotation[1]) < 1e-2
    assert abs((b.results.BMt - )/b.results.BMt) < 1e-2
    assert abs((b.results.GMt - )/b.results.GMt) < 1e-2
    assert abs((b.results.Ixx - )/b.results.Ixx) < 1e-2
    assert abs((b.results.Iyy - )/b.results.Iyy) < 1e-2

    b.set_waterplane(20, -0.876867066127072, -199.411260684941, cor=np.array([5872.8, -3737.5, 0.0]))
    b.calculate_results()
    assert abs((b.results.volume - 2898429049.33231)/b.results.volume) < 1e-2
    assert abs((b.results.wetted_surface_area - 17352704.3346527)/b.results.wetted_surface_area) < 1e-2
    assert abs((b.results.waterplane_area - )/b.results.waterplane_area) < 1e-2
    assert abs((b.results.volume_centroid[0]-)/b.results.volume_centroid[0]) < 1e-2
    assert abs((b.results.volume_centroid[1] - )/b.results.volume_centroid[1]) < 1e-2
    assert abs((b.results.volume_centroid[2] - )/b.results.volume_centroid[2]) < 1e-2
    assert abs((b.results.centre_of_flotation[0] - )/b.results.centre_of_flotation[0]) < 1e-2
    assert abs((b.results.centre_of_flotation[1] - )/b.results.centre_of_flotation[1]) < 1e-2
    assert abs((b.results.BMt - )/b.results.BMt) < 1e-2
    assert abs((b.results.GMt - )/b.results.GMt) < 1e-2
    assert abs((b.results.Ixx - )/b.results.Ixx) < 1e-2
    assert abs((b.results.Iyy - )/b.results.Iyy) < 1e-2
    
    b.set_waterplane(40, -0.727982909099343, -151.565121781228, cor=np.array([5872.8, -3737.5, 0.0]))
    b.calculate_results()
    assert abs((b.results.volume - 2906122622.31428)/b.results.volume) < 1e-2
    assert abs((b.results.wetted_surface_area - 17352704.3346527)/b.results.wetted_surface_area) < 1e-2
    assert abs((b.results.waterplane_area - )/b.results.waterplane_area) < 1e-2
    assert abs((b.results.volume_centroid[0]-)/b.results.volume_centroid[0]) < 1e-2
    assert abs((b.results.volume_centroid[1] - )/b.results.volume_centroid[1]) < 1e-2
    assert abs((b.results.volume_centroid[2] - )/b.results.volume_centroid[2]) < 1e-2
    assert abs((b.results.centre_of_flotation[0] - )/b.results.centre_of_flotation[0]) < 1e-2
    assert abs((b.results.centre_of_flotation[1] - )/b.results.centre_of_flotation[1]) < 1e-2
    assert abs((b.results.BMt - )/b.results.BMt) < 1e-2
    assert abs((b.results.GMt - )/b.results.GMt) < 1e-2
    assert abs((b.results.Ixx - )/b.results.Ixx) < 1e-2
    assert abs((b.results.Iyy - )/b.results.Iyy) < 1e-2

    b.set_waterplane(30, -0.869084284965612, -181.227615044623, cor=np.array([5872.8, -3737.5, 0.0]))
    b.calculate_results()
    assert abs((b.results.volume - 2904292154.19391)/b.results.volume) < 1e-2
    assert abs((b.results.wetted_surface_area - 17352704.3346527)/b.results.wetted_surface_area) < 1e-2
    assert abs((b.results.waterplane_area - )/b.results.waterplane_area) < 1e-2
    assert abs((b.results.volume_centroid[0]-)/b.results.volume_centroid[0]) < 1e-2
    assert abs((b.results.volume_centroid[1] - )/b.results.volume_centroid[1]) < 1e-2
    assert abs((b.results.volume_centroid[2] - )/b.results.volume_centroid[2]) < 1e-2
    assert abs((b.results.centre_of_flotation[0] - )/b.results.centre_of_flotation[0]) < 1e-2
    assert abs((b.results.centre_of_flotation[1] - )/b.results.centre_of_flotation[1]) < 1e-2
    assert abs((b.results.BMt - )/b.results.BMt) < 1e-2
    assert abs((b.results.GMt - )/b.results.GMt) < 1e-2
    assert abs((b.results.Ixx - )/b.results.Ixx) < 1e-2
    assert abs((b.results.Iyy - )/b.results.Iyy) < 1e-2
    """


def test_cube_moments():
    h = BuoyancyModel()
    h.set_weight_force(
        "cube",
        np.array(
            [
                2.0,
                2.0,
                2.0,
            ]
        ),
        8,
    )
    h.g = 1
    h.water_density = 1
    h.meshes["cube"] = close_ends(
        mirror_uv(
            np.flip(
                np.array(
                    [
                        [[1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
                        [[1, 1, 1], [-1, 1, 1], [-1, 0, 1]],
                        [[1, 1, -1], [-1, 1, -1], [-1, 0, -1]],
                        [[1, 0, -1], [-1, 0, -1], [-1, 0, -1]],
                    ],
                    dtype="float64",
                ),
                0,
            )
        )
    )
    h.active_mesh["cube"] = True
    h.show_mesh["cube"] = True
    h.local_position["cube"] = np.array([1.0, 1.0, 1.0])
    h.local_rotation["cube"] = np.array([0.0, 0.0, 0.0])
    h.calculate_transformation()

    h.heel = 0
    h.trim = 0
    h.waterplane_origin = np.array([0.0, 0.0, 0.0])
    h.calculate_results()

    assert h.results.force_earth[2] == -8.0
    assert h.results.moment_earth[0] == -16.0
    assert h.results.moment_earth[1] == 16.0

    h.heel = 0
    h.trim = 0
    h.waterplane_origin = np.array([0.0, 0.0, 2.0])
    h.calculate_results()

    assert h.results.force_earth[2] == 0.0
    assert h.results.moment_earth[0] == -8.0
    assert h.results.moment_earth[1].round(decimals=5) == 8.0

    assert np.all(h.results.force == h.results.force_earth)
    assert np.all(h.results.moment == h.results.moment_earth)

    h.heel = 0
    h.trim = 45
    h.waterplane_origin = np.array([0.0, 0.0, 0.0])
    h.calculate_results()

    assert h.results.force_earth[2].round(decimals=5) == -4.0
    assert abs(h.results.moment_earth[0] + 12.0) < 1e-8
    assert abs(h.results.moment_earth[1] - 6 * np.sqrt(8)) < 1e-8

    h.heel = -45
    h.trim = 0
    h.waterplane_origin = np.array([0.0, 0.0, 0.0])
    h.calculate_results()

    assert h.results.force_earth[2].round(decimals=5) == -4.0
    assert abs(h.results.moment_earth[1] - 12.0) < 1e-8
    assert abs(h.results.moment_earth[0] + 6 * np.sqrt(8)) < 1e-8

    h.heel = -90
    h.trim = 90
    h.waterplane_origin = np.array([0.0, 0.0, 0.0])
    h.calculate_results()

    assert h.results.force_earth[2].round(decimals=5) == 0.0
    assert abs(h.results.moment_earth[1] + 8.0) < 1e-8
    assert abs(h.results.moment_earth[0] + 8.0) < 1e-8


def set_waterplane():
    b = BuoyancyModel()
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
    if pv:
        b.set_waterplane(0, 0, 0, 0)
        b.calculate_results()
        p = pv.Plotter()
        for mesh in b.plot_transformed():
            p.add_mesh(mesh)
        p.add_mesh(b.plot_water_plane())
        p.show_grid()
        p.show()

        b.set_waterplane(90, 0, 0, 0)
        b.calculate_results()
        p = pv.Plotter()
        for mesh in b.plot_transformed():
            p.add_mesh(mesh)
        p.add_mesh(b.plot_water_plane())
        p.show_grid()
        p.show()

        b.set_waterplane(0, 90, 0, 0)
        b.calculate_results()
        p = pv.Plotter()
        for mesh in b.plot_transformed():
            p.add_mesh(mesh)
        p.add_mesh(b.plot_water_plane())
        p.show_grid()
        p.show()

        b.set_waterplane(0, 0, 90, 0)
        b.calculate_results()
        p = pv.Plotter()
        for mesh in b.plot_transformed():
            p.add_mesh(mesh)
        p.add_mesh(b.plot_water_plane())
        p.show_grid()
        p.show()

        b.set_waterplane(0, 0, 0, 1000)
        b.calculate_results()
        p = pv.Plotter()
        for mesh in b.plot_transformed():
            p.add_mesh(mesh)
        p.add_mesh(b.plot_water_plane())
        p.show_grid()
        p.show()
        b.set_waterplane(45, 45, 0, 1000)
        b.calculate_results()
        p = pv.Plotter()
        for mesh in b.plot_transformed():
            p.add_mesh(mesh)
        p.add_mesh(b.plot_water_plane())
        p.show_grid()
        p.show()
        print(b.results.force_earth)
        print(b.results.moment_earth)

        b.set_waterplane(0, 10, 0, 1000)
        b.calculate_results()
        p = pv.Plotter()
        for mesh in b.plot_transformed():
            p.add_mesh(mesh)
        p.add_mesh(b.plot_water_plane())
        p.show_grid()
        p.show()
        b.set_waterplane(45, 45, 90, 1000)
        b.calculate_results()
        p = pv.Plotter()
        for mesh in b.plot_transformed():
            p.add_mesh(mesh)
        p.add_mesh(b.plot_water_plane())
        p.show_grid()
        p.show()
        print(b.results.force_earth)
        print(b.results.moment_earth)


if __name__ == "__main__":
    test_cube_moments()
