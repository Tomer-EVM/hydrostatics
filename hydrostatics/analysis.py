# print("Started")
from hydrostatics.mesh_processing import close_ends, mirror_uv
from hydrostatics.models import BuoyancyModel
from hydrostatics.optimize import iterative, iterative_multidimensional
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


def grid(model, heel=(-180, 180), trim=(-180, 180), resolution=20):
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
    results: list[dict] = []
    for h in np.linspace(heel[0], heel[1], resolution):
        for t in np.linspace(trim[0], trim[1], resolution):
            model.heel = h
            model.trim = t
            model.waterplane_origin = c
            model.calculate_results()
            iterative_multidimensional(
                model, selected=(False, False, True), max_iter=100, max_time=10
            )
            results.append(model.results.dict())

    return results


def analyse_trim(model: BuoyancyModel, trim=(-180, 180), resolution=20):
    c = get_centroid(model)
    results: list[dict] = []
    angles: list[float] = []
    for t in np.linspace(trim[0], trim[1], resolution):
        model.trim = t
        model.waterplane_origin = c
        model.calculate_results()
        iterative_multidimensional(
            model, selected=(True, False, True), max_iter=1000, max_time=10
        )
        results.append(model.results.dict())

    return results


def analyse_heel(model: BuoyancyModel, heel=(-180, 180), resolution=20):
    c = get_centroid(model)
    results: list[dict] = []
    for h in np.linspace(heel[0], heel[1], resolution):
        model.heel = h
        model.waterplane_origin = c
        model.calculate_results()
        iterative_multidimensional(
            model, selected=(False, True, True), max_iter=1000, max_time=10
        )
        results.append(model.results.dict())

    return results
