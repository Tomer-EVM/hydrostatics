# print("Started")
from hydrostatics.mesh_processing import close_ends, mirror_uv
from hydrostatics.models import BuoyancyModel
from hydrostatics.optimize import iterative
import numpy as np
from copy import copy

# print("done")


def get_centroid(model):
    """Returns the volume centroid of the entire model

    Parameters
    ----------
    model : BuoyancyModel
    """
    i = 0
    while model.results.state != "Underwater" and i < 100:
        model.heel = 0
        model.trim = 0
        model.waterplane_origin += np.array([0, 0, 100000])
        model.calculate_results()
        i += 1
    return model.results.volume_centroid


def sink(model):
    """Moves the boat to force equilibrium, ignoring moments

    Parameters
    ----------
    model : BuoyancyModel
    """
    iterative(model, selected=(False, False, True), max_iter=100, max_time=10)


def grid(model, heel=(-180, 180), trim=(-180, 180), resolution=(20, 20)):
    """Produces a grid of results for every trim and heel value

    Sinks the boat to force equilibrium at each iteration

    Parameters
    ----------
    model : BuoyancyModel
    heel : (float, float)
        The range of heel values for the grid
    trim : (float, float)
        The range of trim values for the grid
    resolution : (int, int)
        The number of trim and heel values to compute results for

    Returns
    -------
    results_grid : list(list(Results))
        A 2D list of Result object for every heel and trim value
    meshgrid : np.array
        A meshgrid for the heel and trim values
    """
    c = get_centroid(model)
    results_grid = []
    for h in np.linspace(heel[0], heel[1], resolution[0]):
        results_grid.append([])
        for t in np.linspace(trim[0], trim[1], resolution[1]):
            model.heel = h
            model.trim = t
            model.waterplane_origin = c
            model.calculate_results()
            sink(model)
            print(model.results.force_earth[2])
            # model.calculate_results()
            results_grid[-1].append(copy(model.results))

    return results_grid, np.meshgrid(
        np.linspace(heel[0], heel[1], resolution[0]),
        np.linspace(trim[0], trim[1], resolution[1]),
    )


if __name__ == "__main__":
    b = BuoyancyModel()
    b.set_weight_force("hull", np.array([5882.4, 590, 2247.4]), 2979 * 9.81)
    b.load_mesh("data/hydrostatics_data/F50HullSTB.dxf")
    b.meshes["F50HullSTB.dxf_0"] = close_ends(mirror_uv(b.meshes["F50HullSTB.dxf_0"]))
    b.calculate_transformation()
    b.calculate_results()

    res, values = grid(b)

    import matplotlib.pylab as plt

    a = np.array([[r.moment_earth[0] for r in l] for l in res], dtype=int)
    b = np.array([[r.moment_earth[1] for r in l] for l in res], dtype=int)
    plt.quiver(values[0], values[1], a, b)
    plt.show()
