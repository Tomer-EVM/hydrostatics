import numpy as np
from hydrostatics.models import BuoyancyModel
from hydrostatics.mesh_processing import close_ends, mirror_uv
from time import time

from hydrostatics.transformations import transformation_matrix

# Add working solvers to this list, to get shown in gui
solvers = []

try:
    import ipopt

    def IpOptSolver(
        model,
        selected=(True, True, True),
        tol=(0.001, 0.001, 0.001, 0.1, 0.1, 0.1),
        max_iter=100,
        max_time=100.0,
        bounds=[[-360, 360], [-360, 360], [-np.Inf, np.Inf]],
    ):
        """Solve hydrostatic equilibrium with IpOpt

        ..deprecated:: 0.0.1
            As the other methods are faster and more reliable, and IpOpt
            is not trivial to install, this may be removed

        Current implementation works with IpOpt installed. However, if it starts too far from the global optima, it finds small local optima and never leaves.
        Could potentially do an initial scatter search, to find better starting points?

        This method is slower than iterative approaches in the best case, but doesn't fall into cycles.

        Parameters
        ----------
        model : BuoyancyModel
            The model containing meshes and weights
        selected : (bool, bool, bool)
            Whether to change a variable (heel, trim, height)
        tol : (float, float, float, float, float, float)
            For heel, trim and height, the difference between values that signifies convergence
            For Mx, My and Fx, the absolute tolerance from 0
            (heel, trim, height, Mx, My, Fz)
        max_iter : int
            The maximum number of iterations to perform
        max_time : float
            The maximum time to take, in seconds
        bounds : array-like
            Upper and lower bounds on the variables
            [[heel LB, heel UB],
            [trim LB, trim  UB],
            [height LB, height UB]]

        Notes
        -----
        Modifies the model in-place
        """

        def obj(x):
            model.set_waterplane(x[0], x[1], 0, x[2], model.waterplane_origin)
            model.calculate_results()
            a = model.R @ np.array([1, 0, 0])
            b = model.R @ np.array([0, 1, 0])
            n = model.R @ np.array([0, 0, 1])
            m = np.Inf
            for mesh in model.transformed.values():
                h = mesh @ n
                if np.min(h) < m:
                    m = np.min(h)
            if m < n @ model.waterplane_origin:
                m = 0
            else:
                m -= n @ model.waterplane_origin
            m += 1
            r = np.array(
                [
                    model.results.volume_centroid.dot(b)
                    - model.waterplane_origin.dot(b),
                    model.results.volume_centroid.dot(a)
                    - model.waterplane_origin.dot(a),
                    model.results.force_earth[2]
                    / (
                        model.g * model.water_density * model.results.waterplane_area
                        + 1
                    )
                    * m,
                ]
            )
            # print(x)
            return r @ r

        return False  # ipopt.minimize_ipopt(obj, [model.heel, model.trim, 0.0], bounds=bounds, options={'disp':5, 'max_iter':max_iter, 'tol':min(tol), 'max_cpu_time':max_time})

    class static:
        """Static equilibrium constraints for constrained IpOpt solver

        Parameters
        ----------
        model : BuoyancyModel
            The model containing meshes and weights
        selected : (bool, bool, bool)
            Whether to change a variable (heel, trim, height)
        tol : (float, float, float, float, float, float)
            For heel, trim and height, the difference between values that signifies convergence
            For Mx, My and Fx, the absolute tolerance from 0
            (heel, trim, height, Mx, My, Fz)
        max_iter : int
            The maximum number of iterations to perform
        max_time : float
            The maximum time to take, in seconds
        bounds : array-like
            Upper and lower bounds on the variables
            [[heel LB, heel UB],
            [trim LB, trim  UB],
            [height LB, height UB]]

        Notes
        -----
        Modifies the model in-place
        """

        def __init__(self, model):
            self.model = model
            self.x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
            self.set_x(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

        def set_x(self, x):
            if not np.all(x == self.x):
                self.model.waterplane_origin = x[:3]
                self.model.heel = x[3]
                self.model.trim = x[4]
                self.model.calculate_results()
                self.x = x

        def objective(self, x):
            self.set_x(x)
            return 0.0

        def gradient(self, x):
            self.set_x(x)
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        def constraints(self, x):
            self.set_x(x)
            if self.model.results.state == "floating":
                return np.array(
                    [
                        self.model.results.force_earth[2],
                        self.model.results.moment_earth[0],
                        self.model.results.moment_earth[1],
                    ]
                )
            else:
                return np.array([self.model.results.force_earth[2], 0.0, 0.0])

        def jacobian(self, x):
            self.set_x(x)
            n = self.model.R @ np.array([0, 0, 1])
            Gz = n @ self.model.results.centre_of_gravity
            Bz = n @ self.model.results.volume_centroid
            return np.array(
                [
                    -self.model.water_density
                    * self.model.g
                    * self.model.results.waterplane_area
                    * n.dot(np.array([1, 0, 0])),
                    -self.model.water_density
                    * self.model.g
                    * self.model.results.waterplane_area
                    * n.dot(np.array([0, 1, 0])),
                    -self.model.water_density
                    * self.model.g
                    * self.model.results.waterplane_area
                    * n.dot(np.array([0, 0, 1])),
                    -self.model.water_density
                    * self.model.g
                    * self.model.results.Ax_origin,
                    self.model.water_density
                    * self.model.g
                    * self.model.results.Ay_origin,
                    -self.model.water_density
                    * self.model.g
                    * self.model.results.Ax_origin
                    * n.dot(np.array([1, 0, 0])),
                    -self.model.water_density
                    * self.model.g
                    * self.model.results.Ax_origin
                    * n.dot(np.array([0, 1, 0])),
                    -self.model.water_density
                    * self.model.g
                    * self.model.results.Ax_origin
                    * n.dot(np.array([0, 0, 1])),
                    (
                        np.linalg.norm(self.model.results.weight_force) * Gz
                        - self.model.water_density
                        * self.model.g
                        * Bz
                        * self.model.results.volume
                        - self.model.water_density
                        * self.model.g
                        * self.model.results.Ixx_origin
                    ),
                    self.model.water_density
                    * self.model.g
                    * self.model.results.Ixy_origin,
                    self.model.water_density
                    * self.model.g
                    * self.model.results.Ay_origin
                    * n.dot(np.array([1, 0, 0])),
                    self.model.water_density
                    * self.model.g
                    * self.model.results.Ay_origin
                    * n.dot(np.array([0, 1, 0])),
                    self.model.water_density
                    * self.model.g
                    * self.model.results.Ay_origin
                    * n.dot(np.array([0, 0, 1])),
                    self.model.water_density
                    * self.model.g
                    * self.model.results.Ixy_origin,
                    (
                        np.linalg.norm(self.model.results.weight_force) * Gz
                        - self.model.water_density
                        * self.model.g
                        * Bz
                        * self.model.results.volume
                        - self.model.water_density
                        * self.model.g
                        * self.model.results.Iyy_origin
                    ),
                ]
            )

    def IpOptConstrained(
        model,
        selected=(True, True, True),
        tol=(0.001, 0.001, 0.001, 0.1, 0.1, 0.1),
        max_iter=100,
        max_time=100.0,
        bounds=[[-np.Inf, np.Inf], [-360, 360], [-360, 360]],
    ):
        """IpOpt solver using constraints to enforce equilibrium

        Warning
        -------
        This needs more work to be functional

        Parameters
        ----------
        model : BuoyancyModel
            The model containing meshes and weights
        selected : (bool, bool, bool)
            Whether to change a variable (heel, trim, height)
        tol : (float, float, float, float, float, float)
            For heel, trim and height, the difference between values that signifies convergence
            For Mx, My and Fx, the absolute tolerance from 0
            (heel, trim, height, Mx, My, Fz)
        max_iter : int
            The maximum number of iterations to perform
        max_time : float
            The maximum time to take, in seconds
        bounds : array-like
            Upper and lower bounds on the variables
            [[heel LB, heel UB],
            [trim LB, trim  UB],
            [height LB, height UB]]
        Notes
        -----
        Modifies the model in-place
        """
        bounds = [bounds[0], bounds[0], bounds[0], bounds[1], bounds[2]]
        p = """ipopt.problem(
            n=5,
            m=3,
            problem_obj=static(model),
            lb=[b[0] for b in bounds],
            ub=[b[1] for b in bounds],
            cl=[-tol[3],-tol[4],-tol[5]],
            cu=[tol[3],tol[4],tol[5]]
        )
        p.addOption('print_level', 5)
        x, info = p.solve(np.array([0.0,0.0,0.0,model.heel, model.trim]))
        print(x)
        print(info)"""

    solvers += [IpOptSolver, IpOptConstrained]
except:
    pass
    # print('IpOpt Not Available')


def iterative(
    model,
    selected=(True, True, True),
    tol=(0.001, 0.001, 0.001, 0.001, 0.001, 0.001),
    max_iter=100,
    max_time=100,
    bounds=[[-360, 360], [-360, 360], [-np.Inf, np.Inf]],
    damping=2,
):
    """Iterative method for solving hydrostatic equilibrium

    Uses the current waterplane area and the current volume change needed to guess a new height
    Uses the distance between the buoancy and weight forces, and the distance to the metacentre, to guess the required change in heel and trim

    Parameters
    ----------
    model : BuoyancyModel
        The model containing meshes and weights
    selected : (bool, bool, bool)
        Whether to change a variable (heel, trim, height)
    tol : (float, float, float, float, float, float)
        For heel, trim and height, the difference between values that signifies convergence
        For Mx, My and Fx, the absolute tolerance from 0
        (heel, trim, height, Mx, My, Fz)
    max_iter : int
        The maximum number of iterations to perform
    max_time : float
        The maximum time to take, in seconds
    bounds : array-like
        Upper and lower bounds on the variables
        [[heel LB, heel UB],
        [trim LB, trim  UB],
        [height LB, height UB]]

    Notes
    -----
    Modifies the model in-place
    """

    heel = model.heel
    trim = model.trim
    height = 0
    cor = model.waterplane_origin

    c = iterative_condition(model, tol, max_iter, max_time)
    i = 0
    delta = [1, 1, 1]
    dHeel = 0
    dTrim = 0
    dHeight = 0
    while c(i, delta):
        i += 1
        model.set_waterplane(heel, trim, 0, height, cor)
        model.calculate_results()
        f = model.results.force_earth
        m = model.results.moment_earth
        if (
            abs(m[0]) * selected[0] < tol[3]
            and abs(m[1]) * selected[1] < tol[4]
            and abs(f[2]) * selected[2] < tol[5]
        ):
            print("Found Solution")
            break

        x_hat = model.R @ np.array([1, 0, 0])
        y_hat = model.R @ np.array([0, 1, 0])
        if model.results.state == "Floating":
            if model.results.waterplane_area > 1e-8:
                if selected[0]:
                    dHeel = (
                        max(
                            min(
                                np.rad2deg(
                                    (
                                        model.results.volume_centroid.dot(y_hat)
                                        - model.results.centre_of_gravity.dot(y_hat)
                                    )
                                    / (model.results.GMt + 1)
                                ),
                                bounds[0][1] - model.heel,
                            ),
                            bounds[0][0] - model.heel,
                        )
                        / damping
                    )
                    heel += dHeel
                else:
                    dHeel = 0

                if selected[1]:
                    dTrim = (
                        -max(
                            min(
                                np.rad2deg(
                                    (
                                        model.results.volume_centroid.dot(x_hat)
                                        - model.results.centre_of_gravity.dot(x_hat)
                                    )
                                    / (model.results.GMl + 1)
                                ),
                                bounds[1][1] + model.trim,
                            ),
                            bounds[1][0] + model.trim,
                        )
                        / damping
                    )
                    trim += dTrim
                else:
                    dTrim = 0

                if selected[2]:
                    dHeight = (
                        f[2]
                        / (
                            model.results.waterplane_area
                            * model.g
                            * model.water_density
                            + 1
                        )
                        / damping
                    )
            else:
                dHeight = (
                    f[2]
                    / (
                        (model.results.bounds[0][1] - model.results.bounds[0][0])
                        * (model.results.bounds[1][1] - model.results.bounds[1][0])
                        * model.g
                        * model.water_density
                        + 1
                    )
                    / damping
                )
                dTrim = 0
                dHeel = 0
        else:
            dHeel = 0
            dTrim = 0
            dHeight = f[2] / (
                (model.results.bounds[0][1] - model.results.bounds[0][0])
                * (model.results.bounds[1][1] - model.results.bounds[1][0])
                * model.g
                * model.water_density
                + 1
            )
            if model.results.state == "Flying":
                dHeight += model.results.depth
            elif model.results.state == "Underwater":
                n = model.R @ np.array([0.0, 0.0, 1.0])
                w_z = n @ model.waterplane_origin
                dHeight += w_z - model.results.bounds[2][1]
            else:
                dHeight = 0

        height = max(min(height + dHeight, bounds[2][1]), bounds[2][0])

        if model.results.state == "Underwater" and f[2] < 0:
            print("No Solution: weight force greater than maximum buoyancy")
            return i

        delta = [dHeight, dHeel, dTrim]
    return i


def iterative_force(
    model,
    selected=(True, True, True),
    tol=(0.1, 0.1, 1.0, 0.1, 0.1, 0.1),
    max_iter=100,
    max_time=100,
    bounds=[[-np.Inf, np.Inf], [-360, 360], [-360, 360]],
    damping=2,
):
    """Basic force based simulator

    Tries to simulate the motion of the boat, ignoring acceleration and velocity.
    Increments distance and angles in the direction of forces and moments
    Somewhat useful to visualise the direction the boat will move.
    Can potentially be used to find static equilibrium, but will take a long time.
    Could think of this as basic gradient descent

    Parameters
    ----------
    model : BuoyancyModel
        The model containing meshes and weights
    selected : (bool, bool, bool)
        Whether to change a variable (heel, trim, height)
    tol : (float, float, float, float, float, float)
        Only uses first three values, as the increment distance
    max_iter : int
        The maximum number of iterations to perform
    max_time : float
        The maximum time the method can take
    bounds : array-like
        NOT IMPLEMENTED
    """
    c = iterative_condition(model, tol, max_iter, max_time)
    i = 0
    cor = model.waterplane_origin
    heel = model.heel
    trim = model.trim
    height = 0.0
    model.set_waterplane(heel, trim, 0, height, cor)
    while c(i, [tol[0], tol[1], tol[2]]):
        i += 1
        model.set_waterplane(heel, trim, 0, height, cor)
        model.calculate_results()
        heel += np.sign(model.results.moment_earth[0]) * tol[0] / damping * selected[0]
        trim += np.sign(model.results.moment_earth[1]) * tol[1] / damping * selected[1]
        height += np.sign(model.results.force_earth[2]) * tol[2] / damping * selected[2]
        print(height)
    return i


def iterative_multidimensional(
    model,
    selected=(True, True, True),
    tol=(0.001, 0.001, 0.001, 0.1, 0.1, 0.1),
    max_iter=100,
    max_time=100,
    bounds=[[-360, 360], [-360, 360], [-np.Inf, np.Inf]],
    damping=3,
):
    """Special implementation of Newtons multidimensional root-finding method

    The Jacobian is approximated using the method described in [1]_.
    Newtons method is slightly modified. We normalise by moving the waterplane origin to
    the centre of flotation each iteration, and then moving it by :math:`\delta z` in the waterplane
    z direction. Heel and trim are modified as normal, except they are bounded between -180 and 180 degrees.
    It also handles cases where the boat is fully immersed or not immersed, and detects when the problem
    is unsolvable.

    Parameters
    ----------
    model : BuoyancyModel
        The model containing meshes and weights
    selected : (bool, bool, bool)
        Whether to change a variable (heel, trim, height)
    tol : (float, float, float, float, float, float)
        For heel, trim and height, the difference between values that signifies convergence
        For Mx, My and Fx, the absolute tolerance from 0
        (heel, trim, height, Mx, My, Fz)
    max_iter : int
        The maximum number of iterations to perform
    max_time : float
        The maximum time to take, in seconds
    bounds : array-like
        Upper and lower bounds on the variables
        [[heel LB, heel UB],
        [trim LB, trim  UB],
        [height LB, height UB]]

    Notes
    -----
    Modifies the model in-place

    .. [1] K. Park, J. Cha, N. Ku, "Nonlinear Hydrostatic Analysis of
        the Floating Structure considering the Large Angle of Inclination,"
        Journal of Marine Science and Technology, Vol 26, No. 2 pp. 264-274, 2018
    """
    y = np.array([0.0, 0.0, 0.0])
    c = iterative_condition(model, tol, max_iter, max_time)
    i = 0
    delta = [1, 1, 1]
    while c(i, delta):
        i += 1
        model.calculate_results()
        n = model.R @ np.array([0, 0, 1])
        Gz = n @ model.results.centre_of_gravity
        Bz = n @ model.results.volume_centroid

        y = np.array(
            [
                model.results.force_earth[2],
                model.results.moment_earth[0],
                model.results.moment_earth[1],
            ]
        )

        if model.results.waterplane_area > 1e-8:
            J = np.array(
                [
                    [
                        -model.water_density * model.g * model.results.waterplane_area,
                        -model.water_density * model.g * model.results.Ax * selected[0],
                        model.water_density * model.g * model.results.Ay * selected[1],
                    ],
                    [
                        -model.water_density * model.g * model.results.Ax * selected[2],
                        (
                            np.linalg.norm(model.results.weight_force) * Gz
                            - model.water_density * model.g * Bz * model.results.volume
                            - model.water_density * model.g * model.results.Ixx
                        ),
                        model.water_density * model.g * model.results.Ixy * selected[1],
                    ],
                    [
                        model.water_density * model.g * model.results.Ay * selected[2],
                        model.water_density * model.g * model.results.Ixy * selected[0],
                        (
                            np.linalg.norm(model.results.weight_force) * Gz
                            - model.water_density * model.g * Bz * model.results.volume
                            - model.water_density * model.g * model.results.Iyy
                        ),
                    ],
                ]
            )
            delta = np.linalg.solve(J, y)  # , assume_a='sym')
            delta[1] = (
                max(
                    min(np.rad2deg(delta[1]), model.heel - bounds[0][0]),
                    model.heel - bounds[0][1],
                )
                * selected[0]
                / damping
            )
            delta[2] = (
                max(
                    min(np.rad2deg(delta[2]), model.trim - bounds[1][0]),
                    model.trim - bounds[1][1],
                )
                * selected[1]
                / damping
            )

            model.heel = max(min(model.heel - delta[1], bounds[0][1]), bounds[0][0])
            model.trim = max(min(model.trim - delta[2], bounds[1][1]), bounds[1][0])

            model.waterplane_origin = (
                model.results.centre_of_flotation
                + max(min(delta[0], bounds[2][1]), bounds[2][0])
                * n
                * selected[2]
                / damping
            )
        else:
            if not selected[2]:
                print("No Solution: Need to change height")
                return i
            dHeight = model.results.force_earth[2] / (
                (model.results.bounds[0][1] - model.results.bounds[0][0])
                * (model.results.bounds[1][1] - model.results.bounds[1][0])
                * model.g
                * model.water_density
                + 1
            )
            if model.results.state == "Flying":
                dHeight += model.results.depth
            elif model.results.state == "Underwater":
                n = model.R @ np.array([0.0, 0.0, 1.0])
                w_z = n @ model.waterplane_origin
                dHeight += w_z - model.results.bounds[2][1]
            delta = np.array([dHeight, 0.0, 0.0])
            model.waterplane_origin = model.waterplane_origin - delta[0] * n
            if y[0] < 0 and model.results.state == "Underwater":
                print("No Solution: weight too heavy")
                return i

    return i


def iterative_condition(model, tol, max_iter, max_time):
    start = time()
    print(
        "| Iteration |     Force |  Moment X |  Moment Y |   Heel |   Trim |   ẟHeel |   ẟTrim |   ẟHeight | State"
    )
    print(
        "|-----------|-----------|-----------|-----------|--------|--------|---------|---------|-----------|-------"
    )
    prev = [0.0, 0.0, 0.0]

    def condition(i=0, delta=None):
        nonlocal prev
        f = [
            model.results.force_earth[2],
            model.results.moment_earth[0],
            model.results.moment_earth[1],
        ]
        x = [model.heel, model.trim]

        if delta is None:
            print(
                f"| {i:9n} | {f[0]:9.2e} | {f[1]:9.2e} | {f[2]:9.2e} | {x[0]:6.1f} | {x[1]:6.1f} |          |          |           | {model.results.state}"
            )
            if (
                abs(f[0] - prev[0]) < tol[3]
                and abs(f[1] - prev[1]) < tol[4]
                and abs(f[2] - prev[2]) < tol[5]
            ):
                print("Converged")
                return False
        else:
            print(
                f"| {i:9n} | {f[0]:9.2e} | {f[1]:9.2e} | {f[2]:9.2e} | {x[0]:6.1f} | {x[1]:6.1f} | {delta[1]:7.1f} | {delta[2]:7.1f} | {delta[0]:9.2e} | {model.results.state}"
            )
            if (
                abs(delta[0]) < tol[0]
                and abs(delta[1]) < tol[1]
                and abs(delta[2]) < tol[2]
                and abs(f[0] - prev[0]) < tol[3]
                and abs(f[1] - prev[1]) < tol[4]
                and abs(f[2] - prev[2]) < tol[5]
            ):
                print("Converged")
                return False
        if time() - start > max_time:
            print("Max Time Reached")
            return False
        if i >= max_iter:
            print("Max Iterations Reached")
            return False

        prev = f
        return True

    return condition


solvers += [iterative, iterative_multidimensional, iterative_force]
