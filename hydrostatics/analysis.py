from typing import List
from hydrostatics.models import BuoyancyModel
import numpy as np

from hydrostatics.optimize import iterative


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


def grid(
    model: BuoyancyModel,
    f=iterative,
    tol=(0.001, 0.001, 0.001, 0.1, 0.1, 0.1),
    max_iter=100,
    max_time=100,
    bounds=[[-360, 360], [-360, 360], [-np.Inf, np.Inf]],
    heel=(-180, 180),
    trim=(-180, 180),
    resolution=20,
):
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
    results_grid : list(Results)
        A list of Result object for every heel and trim value
    """
    c = get_centroid(model)
    results: List[dict] = []
    for h in np.linspace(heel[0], heel[1], resolution):
        for t in np.linspace(trim[0], trim[1], resolution):
            model.heel = h
            model.trim = t
            model.waterplane_origin = c
            model.calculate_results()
            f(
                model,
                selected=(False, False, True),
                bounds=bounds,
                tol=tol,
                max_iter=max_iter,
                max_time=max_time,
            )
            results.append(model.results.dict())

    return results


def analyse_trim(
    model: BuoyancyModel,
    f=iterative,
    tol=(0.001, 0.001, 0.001, 0.1, 0.1, 0.1),
    max_iter=100,
    max_time=100,
    bounds=[[-360, 360], [-360, 360], [-np.Inf, np.Inf]],
    trim=(-180, 180),
    resolution=20,
):
    c = get_centroid(model)
    results: List[dict] = []
    for t in np.linspace(trim[0], trim[1], resolution):
        model.trim = t
        model.waterplane_origin = c
        model.calculate_results()
        f(
            model,
            selected=(True, False, True),
            bounds=bounds,
            tol=tol,
            max_iter=max_iter,
            max_time=max_time,
        )
        results.append(model.results.dict())

    return results


def analyse_trim_fixed_heel(
    model: BuoyancyModel,
    f=iterative,
    tol=(0.001, 0.001, 0.001, 0.1, 0.1, 0.1),
    max_iter=100,
    max_time=100,
    bounds=[[-360, 360], [-360, 360], [-np.Inf, np.Inf]],
    trim=(-180, 180),
    resolution=20,
):
    c = get_centroid(model)
    results: List[dict] = []
    for t in np.linspace(trim[0], trim[1], resolution):
        model.trim = t
        model.waterplane_origin = c
        model.calculate_results()
        f(
            model,
            selected=(False, False, True),
            bounds=bounds,
            tol=tol,
            max_iter=max_iter,
            max_time=max_time,
        )
        results.append(model.results.dict())

    return results


def analyse_heel(
    model: BuoyancyModel,
    f=iterative,
    tol=(0.001, 0.001, 0.001, 0.1, 0.1, 0.1),
    max_iter=100,
    max_time=100,
    bounds=[[-360, 360], [-360, 360], [-np.Inf, np.Inf]],
    heel=(-180, 180),
    resolution=20,
):
    c = get_centroid(model)
    results: List[dict] = []
    for h in np.linspace(heel[0], heel[1], resolution):
        model.heel = h
        model.waterplane_origin = c
        model.calculate_results()
        f(
            model,
            selected=(False, True, True),
            bounds=bounds,
            tol=tol,
            max_iter=max_iter,
            max_time=max_time,
        )
        results.append(model.results.dict())

    return results


def analyse_heel_fixed_trim(
    model: BuoyancyModel,
    f=iterative,
    tol=(0.001, 0.001, 0.001, 0.1, 0.1, 0.1),
    max_iter=100,
    max_time=100,
    bounds=[[-360, 360], [-360, 360], [-np.Inf, np.Inf]],
    heel=(-180, 180),
    resolution=20,
):
    c = get_centroid(model)
    results: List[dict] = []
    for h in np.linspace(heel[0], heel[1], resolution):
        model.heel = h
        model.waterplane_origin = c
        model.calculate_results()
        f(
            model,
            selected=(False, False, True),
            bounds=bounds,
            tol=tol,
            max_iter=max_iter,
            max_time=max_time,
        )
        results.append(model.results.dict())

    return results
