from time import time
import os.path
import pickle
import sys

from hydrostatics.mesh_processing import *
from hydrostatics.transformations import *

try:
    import pyvista as pv
except:
    pass

import numpy as np
import trimesh


class BuoyancyModel:
    """Computes hydrostatic forces on objects

    Each mesh is represented by a NxMx3 array, with uv coordinates for the first two dimensions and [x,y,z] for the third.

    Each weight force is represented by a position and magnitude pair

    Attributes
    ----------
    meshes : dict
    weight_forces : dict
    local_position : dict
    local_rotation : dict
    transformed : dict
    active_mesh : dict
    active_weight : dict
    show_mesh : dict
    show_weight : dict
    heel : float
    trim : float
    waterplane_origin : array-like
    R : array-like
    faces_below_water : dict
    water_edge : dict
    water_density : float
    g : float
    results : Results
    """

    if True:  # Init and IO

        def __init__(self):
            self.meshes = {}  # Raw meshes read from file
            self.weight_forces = {}

            self.local_position = {}
            self.local_rotation = {}
            self.transformed = {}  # Meshes transformed by local transforms
            self.active_mesh = {}
            self.active_weight = {}

            self.show_mesh = {}
            self.show_weight = {}

            self.heel = 0
            self.trim = 0
            self.leeway = 0
            # self.height = 0
            # self.cor = np.array([0, 0, 0]) # Centre of rotation
            self.waterplane_origin = np.array([0.0, 0.0, 0.0])
            self.body_origin = np.array([0.0, 0.0, 0.0])
            # self.cutting_plane = np.array([0, 0, 0])
            self.R = np.eye(3)

            self.faces_below_water = {}
            self.water_edge = {}

            self.water_density = 1025.0 * 1e-9
            self.g = 9.81

            self.results = self.Results()

            self.include_interior = True

        def set_weight_force(self, name: str, position: np.array, value: float):
            """Fills out the relevant dicts with a new weight force

            Parameters
            ----------
            name : str
                The name of the weight
            position : array-like
                The position in 3D of the weight
            value : float
                The weight in Newtons
            """
            self.weight_forces[name] = [position, value]
            self.active_weight[name] = True
            self.show_weight[name] = True
            self.run()

        # TODO: Check reading method always works
        def load_mesh(self, filename: str, shape=None):
            """Read DXF file data

            Imports the file data, converts to a UV mesh, and sets default mesh position and properties.
            Recalculates hydrostatic results after adding.

            Parameters
            ----------
            filename : str
                The path to the dxf file
            shape : (int, int), optional
                The shape of the UV array to represent the dxf data.
                If this is not given, the closest shape to a square is chosen.
            """
            ext = os.path.splitext(filename)[1]
            names = []
            if ext == ".dxf":
                meshes = read_data(filename)
                name = os.path.basename(filename)
                for i, mesh in enumerate(meshes):
                    if shape is None or shape[0] * shape[1] != mesh.shape[0]:
                        shape = get_closest_divisors(mesh.shape[0])
                    self.meshes[name + f"_{i}"] = convert_to_uv(mesh, shape)
                    self.local_position[name + f"_{i}"] = np.array([0, 0, 0])
                    self.local_rotation[name + f"_{i}"] = np.array([0, 0, 0])
                    self.active_mesh[name + f"_{i}"] = True
                    self.show_mesh[name + f"_{i}"] = True
                    names.append(name + f"_{i}")
                self.calculate_transformation()

            elif ext == ".npy":
                name = os.path.basename(filename)
                self.meshes[name] = load_uv(filename)
                self.local_position[name] = np.array([0, 0, 0])
                self.local_rotation[name] = np.array([0, 0, 0])
                self.calculate_transformation()
                self.active_mesh[name] = True
                self.show_mesh[name] = True
                names.append(name)
            else:
                try:
                    name = os.path.basename(filename)
                    res = trimesh.load_mesh(filename)
                    if isinstance(res, trimesh.Scene):
                        for i, mesh in enumerate(res.geometry.values()):
                            self.meshes[name + f"_{i}"] = mesh
                            self.local_position[name + f"_{i}"] = np.array([0, 0, 0])
                            self.local_rotation[name + f"_{i}"] = np.array([0, 0, 0])
                            self.active_mesh[name + f"_{i}"] = True
                            self.show_mesh[name + f"_{i}"] = True
                            names.append(name + f"_{i}")

                    elif isinstance(res, trimesh.Trimesh):
                        self.meshes[name] = res
                        self.local_position[name] = np.array([0, 0, 0])
                        self.local_rotation[name] = np.array([0, 0, 0])
                        self.active_mesh[name] = True
                        self.show_mesh[name] = True
                        names.append(name)
                    self.calculate_transformation()
                except:
                    print("File format not supported")
            self.calculate_results()
            return names

        def save(self, filename: str):
            """Saves the buoyancy model

            A simple wrapper over pickle that saves this model to a .hydro binary file in pickle format

            Parameters
            ----------
            filename : str
                The file path to save to
            """
            if os.path.splitext(filename)[1] == "":
                filename = filename + ".hydro"
            with open(filename, "wb") as f:
                pickle.dump(self, f)

    if True:  # Transformations

        def mirror(self, name):
            """Attempts to mirror the specified mesh about the x-axis

            Parameters
            ----------
            name : str
                The name of the mesh

            Warning
            -------
            This operation will only work for UV meshes
            """
            if isinstance(self.meshes[name], np.ndarray):
                self.meshes[name] = mirror_uv(self.meshes[name])
                self.calculate_transformation()

        def close_ends(self, name):
            """Attempts to close open ends of the mesh

            Parameters
            ----------
            name : str
                The name of the mesh

            Warning
            -------
            This operation will only work for UV meshes
            """
            if isinstance(self.meshes[name], np.ndarray):
                self.meshes[name] = close_ends(self.meshes[name])
                self.calculate_transformation()

        def calculate_transformation(self):
            """Computes new mesh locations based on local position and rotation values

            Stores results in transformed
            """
            for name, mesh in self.meshes.items():
                if isinstance(mesh, np.ndarray):
                    r = transformation_matrix(
                        *self.local_rotation[name]
                    )  # Rotational transformation matrix
                    # Transform rotations
                    """
                    for i in range(self.transformed[name].shape[0]):
                        for j in range(self.transformed[name].shape[1]):
                            self.transformed[i,j,:] = R.dot(self.transformed[i,j,:])
                    """
                    temp = (
                        mesh @ r.T
                    )  # This should be equivalent to above code, but faster as handled by numpy

                    # Transform heights
                    temp += np.array(
                        [
                            [self.local_position[name] for j in range(mesh.shape[1])]
                            for i in range(mesh.shape[0])
                        ]
                    )
                    temp = uv_to_trimesh(temp)
                else:
                    r = trimesh.transformations.euler_matrix(
                        self.local_rotation[name][0],
                        self.local_rotation[name][1],
                        self.local_rotation[name][2],
                        "rxyz",
                    )
                    t = trimesh.transformations.translation_matrix(
                        self.local_position[name]
                    )
                    temp = mesh.copy().apply_transform(r).apply_transform(t)
                self.transformed[name] = Mesh(
                    np.array(temp.vertices, dtype=float),
                    np.array(temp.faces, dtype=int),
                )

        def cut(self, update_R=True):
            """Cuts the meshes at the waterline

            Stores faces in faces_below_water and edges in water_edge
            """

            self.faces_below_water = {}
            self.water_edge = {}

            if update_R:
                self.R = transformation_matrix(-self.heel, -self.trim, -self.leeway)
            normal = self.R @ np.array([0, 0, 1])

            for name, mesh in self.transformed.items():
                if self.active_mesh[name]:
                    f, e = mesh.cut(
                        self.waterplane_origin,
                        normal,
                        include_interior=self.include_interior,
                    )
                    self.faces_below_water[name] = f
                    self.water_edge[name] = e

    if True:  # Force calculation

        def set_waterplane(
            self, heel, trim, leeway, height, cor=np.array([0.0, 0.0, 0.0])
        ):
            """Sets the position of the waterplane using heel, trim and height

            Parameters
            ----------
            heel : float
                The heel, or roll, of the boat
            trim : float
                The trim, or pitch, of the boat
            height : float
                The height to raise the boat by
            cor : array-like, optional
                The position to take rotations about. Defaults to the origin
            """
            self.heel = heel
            self.trim = trim
            self.leeway = leeway
            self.R = transformation_matrix(-self.heel, -self.trim, -self.leeway)
            self.waterplane_origin = self.R @ np.array([0, 0, -height]) + cor

        def set_waterplane_vpp(
            self,
            heel,
            trim,
            leeway,
            height,
            body_origin=np.array([0.0, 0.0, 0.0]),
            waterplane_origin=np.array([0.0, 0.0, 0.0]),
        ):
            """Sets the position of the waterplane using heel, trim and height, corrected for vpp

            Parameters
            ----------
            heel : float
                The heel, or roll, of the boat
            trim : float
                The trim, or pitch, of the boat
            height : float
                The height to raise the boat by
            body_origin : np.array
                The initial position of the body origin in body coordinates
            waterplane_origin : np.array
                The initial position of the waterplane origin in body coordinates
            """
            self.heel = heel
            self.trim = -trim
            self.leeway = -leeway
            self.R = transformation_matrix(-self.heel, self.trim, self.leeway)
            self.waterplane_origin = self.R @ (
                waterplane_origin - np.array([0, 0, -height])
            )
            self.body_origin = body_origin

        def get_buoyancy(self):
            """Calculates the buoyancy forces and moments of active meshes"""
            self.cut()

            volumes = []
            centroids = []
            for name, mesh in self.faces_below_water.items():
                volume, centroid = mesh.volume_properties(self.waterplane_origin)
                volumes.append(volume)
                centroids.append(centroid)
                self.results.mesh_results[name].volume = volume
                self.results.mesh_results[name].volume_centroid = centroid
            overall_volume = sum(volumes)
            if overall_volume != 0:
                overall_centroid = (
                    np.array(volumes) @ np.array(centroids) / overall_volume
                )
            else:
                overall_centroid = np.array([0.0, 0.0, 0.0])
            self.results.volume = overall_volume
            self.results.volume_centroid = overall_centroid

            for res in self.results.mesh_results.values():
                res.force_earth = np.array(
                    [0.0, 0.0, self.water_density * self.g * res.volume]
                )
                res.force = self.R @ res.force_earth
                res.moment_earth = np.cross(
                    np.linalg.solve(
                        self.R, res.volume_centroid - self.waterplane_origin
                    ),
                    res.force_earth,
                )
                res.moment = np.cross(
                    res.volume_centroid - self.waterplane_origin, res.force
                )

            force = self.R @ np.array(
                [0.0, 0.0, self.water_density * self.g * overall_volume]
            )
            moment = np.cross(overall_centroid - self.waterplane_origin, force)

            self.results.buoyancy_force = force
            self.results.buoyancy_moment = moment

            return force, moment

        def run(self, reference="body"):
            """Calculate forces and moments at the current state

            Parameters
            ----------
            Reference: str
                The reference coordinate system.
                Can be 'body', 'earth' or a 3D rotation matrix from earth coordinates.

            Returns
            -------
            forces : array-like
                The resultant forces of all meshes and weights in the current reference frame.
            moments : array-like
                The resultant moments of all meshes and weights in the current reference frame.
            """

            self.results.reset()
            self.results.heel = self.heel
            self.results.trim = self.trim
            for name, active in self.active_mesh.items():
                self.results.mesh_results[name] = self.results.MeshResult()
            for name, active in self.active_weight.items():
                self.results.weight_results[name] = self.results.WeightResult()

            b_f, b_m = self.get_buoyancy()

            for name, w in self.weight_forces.items():
                if self.active_weight[name]:
                    self.results.weight_results[name].force_earth = np.array(
                        [0.0, 0.0, -w[1]]
                    )
                    self.results.weight_results[name].force = self.R @ np.array(
                        [0.0, 0.0, -w[1]]
                    )
                    self.results.weight_results[name].moment = np.cross(
                        w[0] - self.waterplane_origin,
                        self.R @ np.array([0.0, 0.0, -w[1]]),
                    )
                    self.results.weight_results[name].moment_earth = np.cross(
                        np.linalg.solve(self.R, w[0] - self.waterplane_origin),
                        np.array([0.0, 0.0, -w[1]]),
                    )
                    self.results.weight_results[name].center_of_gravity = w[0]

            w_f = self.R @ np.array(
                [
                    0.0,
                    0.0,
                    -sum(
                        w[1]
                        for name, w in self.weight_forces.items()
                        if self.active_weight[name]
                    ),
                ]
            )
            w_m = sum(
                np.cross(
                    w[0] - self.waterplane_origin, self.R @ np.array([0.0, 0.0, -w[1]])
                )
                for w in self.weight_forces.values()
            ) + np.array([0.0, 0.0, 0.0])

            self.results.current = True
            self.results.partial = True
            self.results.force = b_f + w_f
            self.results.moment = b_m + w_m
            self.results.force_earth = np.linalg.solve(
                self.R, b_f + w_f
            )  # sum(res.force_earth for res in self.results.mesh_results.values()) + sum(res.force_earth for res in self.results.weight_results.values()) + np.array([0.,0.,0.])
            self.results.moment_earth = (
                sum(res.moment_earth for res in self.results.mesh_results.values())
                + sum(res.moment_earth for res in self.results.weight_results.values())
                + np.array([0.0, 0.0, 0.0])
            )
            self.results.weight_force = w_f
            self.results.weight_moment = w_m

            if reference == "body":
                return b_f + w_f, b_m + w_m
            elif reference == "earth":
                return self.results.force_earth, self.results.moment_earth
            else:
                # Assume reference is a rotation matrix from earth coordinates
                return (
                    reference @ self.results.force_earth,
                    reference @ self.results.moment_earth,
                )

        def run_vpp(
            self,
            heel,
            trim,
            leeway,
            height,
            body_origin=np.array([0.0, 0.0, 0.0]),
            waterplane_origin=np.array([0.0, 0.0, 0.0]),
        ):
            """Calculates the buoyancy forces and moments of active meshes

            Parameters
            ----------
            heel : float
                The heel, or roll, of the boat
            trim : float
                The trim, or pitch, of the boat
            height : float
                The height to raise the boat by
            body_origin : np.array
                The initial position of the body origin in body coordinates
            waterplane_origin : np.array
                The initial position of the waterplane origin in body coordinates
            """

            self.heel = heel
            self.trim = -trim
            self.leeway = -leeway
            self.R = transformation_matrix(-self.heel, self.trim, self.leeway)
            self.waterplane_origin = self.R @ (
                waterplane_origin - np.array([0, 0, -height])
            )
            self.body_origin = body_origin

            self.cut(update_R=False)

            volumes = np.zeros(len(self.faces_below_water))
            centroids = np.zeros((len(self.faces_below_water), 3))
            for i, mesh in enumerate(self.faces_below_water.values()):
                volume, centroid = mesh.volume_properties(self.waterplane_origin)
                volumes[i] = volume
                centroids[i, :] = centroid

            overall_volume = sum(volumes)
            if overall_volume != 0:
                overall_centroid = volumes @ centroids / overall_volume
            else:
                overall_centroid = np.array([0.0, 0.0, 0.0])

            correction = np.array([1, -1, -1])  # Flip directions of y and z
            force = correction * (
                self.R
                @ np.array([0.0, 0.0, self.water_density * self.g * overall_volume])
            )
            moment = np.cross(
                correction * (overall_centroid - self.body_origin) / 1000, force
            )

            return force, moment

    if True:  # Full Results

        def bounds(self):
            """Calculates a bounding box and immersion state

            Bounding box is aligned with the waterplane.
            Possible state values are: 'Flying', 'Floating', and 'Underwater'.

            Returns
            -------
            np.array
                A 3x2 array of min and max values, in the waterplanes x,y and z directions
            """
            x = self.R @ np.array([1.0, 0.0, 0.0])
            y = self.R @ np.array([0.0, 1.0, 0.0])
            z = self.R @ np.array([0.0, 0.0, 1.0])

            waterplane_z = z.dot(self.waterplane_origin)

            bounds = np.array([[np.Inf, -np.Inf], [np.Inf, -np.Inf], [np.Inf, -np.Inf]])
            for name, mesh in self.transformed.items():
                xs = mesh.vertices @ x
                ys = mesh.vertices @ y
                zs = mesh.vertices @ z
                bounds[0, 0] = min(np.min(xs), bounds[0, 0])
                bounds[0, 1] = max(np.max(xs), bounds[0, 1])
                bounds[1, 0] = min(np.min(ys), bounds[1, 0])
                bounds[1, 1] = max(np.max(ys), bounds[1, 1])
                bounds[2, 0] = min(np.min(zs), bounds[2, 0])
                bounds[2, 1] = max(np.max(zs), bounds[2, 1])

                self.results.mesh_results[name].bounds = np.array(
                    [
                        [np.min(xs), np.max(xs)],
                        [np.min(ys), np.max(ys)],
                        [np.min(zs), np.max(zs)],
                    ]
                )
                self.results.mesh_results[name].depth = (
                    waterplane_z - self.results.mesh_results[name].bounds[2, 0]
                )

            self.results.bounds = bounds
            self.results.depth = waterplane_z - self.results.bounds[2, 0]

            if self.results.volume < 1e-9:
                self.results.state = "Flying"
            elif self.results.bounds[2, 1] - waterplane_z < 0:
                self.results.state = "Underwater"
            else:
                self.results.state = "Floating"

            for res in self.results.mesh_results.values():
                if res.volume < 1e-9:
                    res.state = "Flying"
                elif res.bounds[2, 1] - waterplane_z < 0:
                    res.state = "Underwater"
                else:
                    res.state = "Floating"

            return bounds

        def waterplane(self):
            """Calculates the area, centroid and moments for the waterplane intersecting with the boat.

            Returns
            -------
            overall_area : float
                The area of the waterplane.
            overall_centroid : np.array
                The centroid of the waterplane in 3D.
            overall_A : np.array
                The first moments of area of the waterplane, Ax and Ay
            overall_I : np.array
                The second moments of area of the waterplane, Ixx, Iyy and Ixy
            """
            overall_area = 0.0
            overall_centroid = np.array([0.0, 0.0, 0.0])
            overall_I = np.array([0.0, 0.0, 0.0])
            overall_A = np.array([0.0, 0.0])

            overall_min_x = np.inf
            overall_max_x = -np.inf
            overall_min_y = np.inf
            overall_max_y = -np.inf

            for name, edges in self.water_edge.items():
                area = 0.0
                centroid = np.array([0.0, 0.0, 0.0])
                min_x = np.inf
                max_x = -np.inf
                min_y = np.inf
                max_y = -np.inf
                edges.x = self.R @ np.array([1.0, 0.0, 0.0])
                edges.y = self.R @ np.array([0.0, 1.0, 0.0])
                area, centroid = edges.area_properties()
                if len(edges.edges) > 0:
                    vertices = edges.vertices[edges.edges.flatten()]
                    xs = vertices @ edges.x
                    ys = vertices @ edges.y
                    min_x = min(xs)
                    min_y = min(ys)
                    max_x = max(xs)
                    max_y = max(ys)

                self.results.mesh_results[name].waterplane_area = area
                self.results.mesh_results[name].centre_of_flotation = centroid
                self.results.mesh_results[name].Lwl = max_x - min_x
                self.results.mesh_results[name].Bwl = max_y - min_y
                self.results.mesh_results[name].Cwp = area / (
                    self.results.mesh_results[name].Lwl
                    * self.results.mesh_results[name].Bwl
                )

                overall_area += area
                overall_centroid += centroid * area

                if max_x > overall_max_x:
                    overall_max_x = max_x
                if min_x < overall_min_x:
                    overall_min_x = min_x
                if max_y > overall_max_y:
                    overall_max_y = max_y
                if min_y < overall_min_y:
                    overall_min_y = min_y

            self.results.waterplane_area = overall_area
            self.results.centre_of_flotation = (
                overall_centroid / overall_area
                if overall_area != 0
                else np.array([0.0, 0.0, 0.0])
            )

            self.results.Lwl = overall_max_x - overall_min_x
            self.results.Bwl = overall_max_y - overall_min_y
            self.results.Cwp = self.results.waterplane_area / (
                self.results.Lwl * self.results.Bwl
            )

            x = self.R @ np.array([1.0, 0.0, 0.0])
            y = self.R @ np.array([0.0, 1.0, 0.0])

            for name, edges in self.water_edge.items():
                I = np.array([0.0, 0.0, 0.0])
                A = np.array([0.0, 0.0])
                edges.centre = self.results.centre_of_flotation
                A += edges.moment_area()
                I += edges.moment_inertia()
                overall_A += A
                overall_I += I
                res = self.results.mesh_results[name]
                res.Ax = A[0]
                res.Ay = A[1]
                res.Ixx = I[0]
                res.Iyy = I[1]
                res.Ixy = I[2]
                res.Ixx_origin = I[0] - res.waterplane_area * (
                    0
                    - y.dot(self.waterplane_origin - self.results.centre_of_flotation)
                    ** 2
                )
                res.Iyy_origin = I[1] - res.waterplane_area * (
                    0
                    - y.dot(self.waterplane_origin - self.results.centre_of_flotation)
                    ** 2
                )
                res.Ixy_origin = I[2] + res.waterplane_area * (
                    x.dot(self.results.centre_of_flotation)
                    * y.dot(self.results.centre_of_flotation)
                    - x.dot(self.results.centre_of_flotation)
                    * y.dot(self.waterplane_origin)
                    - y.dot(self.results.centre_of_flotation)
                    * x.dot(self.waterplane_origin)
                )

                res.Ax_origin = A[0] + res.waterplane_area * y.dot(
                    self.results.centre_of_flotation - self.waterplane_origin
                )
                res.Ay_origin = A[1] + res.waterplane_area * y.dot(
                    self.results.centre_of_flotation - self.waterplane_origin
                )

                if (
                    -1e-4 < (I[0] - I[1]) ** 2 / 4 + I[2] < 0
                ):  # Catch numerical errors on symmetric cross-section
                    res.Iu = (res.Ixx + res.Iyy) / 2
                    res.Iv = (res.Ixx + res.Iyy) / 2
                else:
                    res.Iu = (res.Ixx + res.Iyy) / 2 + np.sqrt(
                        (res.Ixx - res.Iyy) ** 2 / 4 + res.Ixy
                    )
                    res.Iv = (res.Ixx + res.Iyy) / 2 - np.sqrt(
                        (res.Ixx - res.Iyy) ** 2 / 4 + res.Ixy
                    )

                if abs(I[2]) > 1e-9:
                    res.theta_principle = np.rad2deg(
                        0.5 * np.arctan2((res.Iyy - res.Ixx), 2 * res.Ixy)
                    )
                else:
                    res.theta_principle = 0.0

            self.results.Ax = overall_A[0]
            self.results.Ay = overall_A[1]

            self.results.Ixx = overall_I[0]
            self.results.Iyy = overall_I[1]
            self.results.Ixy = overall_I[2]

            self.results.Ixx_origin = overall_I[0] - overall_area * (
                0
                - y.dot(self.waterplane_origin - self.results.centre_of_flotation) ** 2
            )
            self.results.Iyy_origin = overall_I[1] - overall_area * (
                0
                - x.dot(self.waterplane_origin - self.results.centre_of_flotation) ** 2
            )
            self.results.Ixy_origin = overall_I[2] + self.results.waterplane_area * (
                x.dot(self.results.centre_of_flotation)
                * y.dot(self.results.centre_of_flotation)
                - x.dot(self.results.centre_of_flotation)
                * y.dot(self.waterplane_origin)
                - y.dot(self.results.centre_of_flotation)
                * x.dot(self.waterplane_origin)
            )

            self.results.Ax_origin = overall_A[0] + overall_area * y.dot(
                self.results.centre_of_flotation - self.waterplane_origin
            )
            self.results.Ay_origin = overall_A[1] + overall_area * x.dot(
                self.results.centre_of_flotation - self.waterplane_origin
            )

            if (
                -1e-4 < (overall_I[0] - overall_I[1]) ** 2 / 4 + overall_I[2] < 0
            ):  # Catch numerical errors on symmetric cross-section
                self.results.Iu = (self.results.Ixx + self.results.Iyy) / 2
                self.results.Iv = (self.results.Ixx + self.results.Iyy) / 2
            else:
                self.results.Iu = (self.results.Ixx + self.results.Iyy) / 2 + np.sqrt(
                    (self.results.Ixx - self.results.Iyy) ** 2 / 4 + self.results.Ixy
                )
                self.results.Iv = (self.results.Ixx + self.results.Iyy) / 2 - np.sqrt(
                    (self.results.Ixx - self.results.Iyy) ** 2 / 4 + self.results.Ixy
                )

            if abs(overall_I[2]) > 1e-9:
                self.results.theta_principle = np.rad2deg(
                    0.5
                    * np.arctan2(
                        (self.results.Iyy - self.results.Ixx), 2 * self.results.Ixy
                    )
                )
            else:
                self.results.theta_principle = 0.0

            return overall_area, overall_centroid, overall_A, overall_I

        def get_centre_of_gravity(self):
            """Finds the centre of gravity from the weights

            Returns
            -------
            np.array
                The cog in 3D
            """
            if (
                len(self.weight_forces) > 0
                and sum(w[1] for w in self.weight_forces.values()) > 0
            ):
                self.results.centre_of_gravity = sum(
                    weight[0] * weight[1] for weight in self.weight_forces.values()
                ) / sum(weight[1] for weight in self.weight_forces.values())
            else:
                self.results.centre_of_gravity = np.array([0.0, 0.0, 0.0])
            return self.results.centre_of_gravity

        def get_wetted_surface_area(self):
            """Calculates the wetted surface area of the mesh

            In other words, the surface area of mesh elements below the waterplane.

            Returns
            -------
            float
                The wetted surface area.
            """
            self.results.wetted_surface_area = 0.0
            for name, mesh in self.faces_below_water.items():
                a = mesh.area()
                self.results.mesh_results[name].wetted_surface_area = a
                self.results.wetted_surface_area += a
            return self.results.wetted_surface_area

        def calculate_results(self):
            """Calculates the full set of results and stores them in self.results

            Most calculations are done in other functions. This function also calculates the
            Metacentre distances, BMt, BMl, GMt and GMl
            """
            self.run()
            self.get_centre_of_gravity()
            self.waterplane()
            self.get_wetted_surface_area()
            self.bounds()
            self.results.BMt = (
                self.results.Ixx / self.results.volume if self.results.volume > 0 else 0
            )
            self.results.BMl = (
                self.results.Iyy / self.results.volume if self.results.volume > 0 else 0
            )
            self.results.GMt = (
                self.results.volume_centroid[2]
                + self.results.BMt
                - self.results.centre_of_gravity[2]
                if self.results.volume > 0
                else 0
            )
            self.results.GMl = (
                self.results.volume_centroid[2]
                + self.results.BMl
                - self.results.centre_of_gravity[2]
                if self.results.volume > 0
                else 0
            )

            self.results.GZ_l = (
                self.results.volume_centroid - self.results.centre_of_gravity
            ) @ (self.R @ np.array([1, 0, 0]))
            self.results.GZ_t = (
                self.results.volume_centroid - self.results.centre_of_gravity
            ) @ (self.R @ np.array([0, 1, 0]))
            self.results.volume_error = (
                self.results.force_earth[2] / self.g / self.water_density
            )

            for res in self.results.mesh_results.values():
                res.BMt = res.Ixx / res.volume if res.volume > 0 else 0
                res.BMl = res.Iyy / res.volume if res.volume > 0 else 0
                res.GMt = (
                    res.volume_centroid[2] + res.BMt - self.results.centre_of_gravity[2]
                    if res.volume > 0
                    else 0
                )
                res.GMl = (
                    res.volume_centroid[2] + res.BMl - self.results.centre_of_gravity[2]
                    if res.volume > 0
                    else 0
                )
                res.GZ_l = (res.volume_centroid - self.results.centre_of_gravity) @ (
                    self.R @ np.array([1, 0, 0])
                )
                res.GZ_t = (res.volume_centroid - self.results.centre_of_gravity) @ (
                    self.R @ np.array([0, 1, 0])
                )

            self.results.current = True
            self.results.partial = False

        class Results:
            """Storage of results

            Attributes
            ----------
            current : bool
                Whether the results are up to date. Not currently used.
            partial : bool
                Whether the results are complete. Changes plain text slightly.
            force : array-like
                The force in body reference in x,y,z
            moment : array-like
                The moment in body reference in x,y,z
            force_earth : array-like
                The force in earth reference in x,y,z
            moment_earth : array-like
                The moment in earth reference in x,y,z
            weight_force : array-like
                The total force of weights, in body reference, in x,y,z
            weight_moment : array-like
                The total mopment of weights, in body reference, in x,y,z
            centre_of_gravity : array-like
                The position of the centre of gravity for all weights
            volume : float
                The total volume displaced
            volume_centroid : array-like
                The centroid of the volume displaced
            buoyancy_force : array-like
                The total bouyancy force from all meshes
            buoyancy_moment : array-like
                The total buoyancy moment from all meshes
            wetted_surface_area :float
                The area of all faces below the waterplane
            waterplane_area : float
                The area of the waterplane
            centre_of_flotation : array-like
                The centroid of the waterplane
            Ax : float
                The first moment of area in x of the waterplane
            Ay : float
                The first moment of area in y of the waterplane
            Ax_origin : float
                The first moment of area in x of the waterplane, taken about the waterplane origin
            Ay_origin : float
                The first moment of area in y of the waterplane, taken about the waterplane origin
            Ixx : float
                The second moment of area in x of the waterplane
            Iyy : float
                The second moment of area in y of the waterplane
            Ixy : float
                The second moment of area in both x and y of the waterplane
            Ixx_origin : float
                The second moment of area in x of the waterplane, taken about the waterplane origin
            Iyy_origin : float
                The second moment of area in y of the waterplane, taken about the waterplane origin
            Ixy_origin : float
                The second moment of area in both x and y of the waterplane, taken about the waterplane origin
            Iu : float
                The first principle second moment of area
            Iv : float
                The second principle second moment of area
            theta_principle : float
                The angle between the original and principle second moments of area, in degrees
            BMt : float
                The transverse distance from the centre of buoyancy to the metacentre
            BMl : float
                The longitudinal distance from the centre of buoyancy to the metacentre
            GMt : float
                The transverse distance from the centre of gravity to the metacentre
            GMl : float
                The longitudinal distance from the centre of gravity to the metacentre
            state : str
                'Flying' if the boat is entirely above the water.
                'Underwater' if the boat is entirely below the water.
                'Floating' if the boat is partially immersed.
            Lwl : float
                Length of waterline
            Bwl : float
                Beam of waterline
            Cwp : float
                The ratio of the waterplane area to Lwl * BWl
            bounds : array-like
                A 3x2 array of min and max values, in the waterplanes x,y and z directions
            depth : float
                The maximum depth of the boat below the waterplane
            GZ_l : float
                The righting arm in the longitudinal direction
            GZ_t : float
                The righting arm in the transverse direction
            volume_error : float
                The amountaf extra displaced volume needed for force equilibrium
            mesh_results : dict
                MeshResult objects for every mesh
            weight_results : dict
                WeightResult objects for every weight
            """

            class MeshResult:
                """Storage for individual mesh results

                Attributes
                ----------
                force : array-like
                    The force in body reference in x,y,z
                moment : array-like
                    The moment in body reference in x,y,z
                force_earth : array-like
                    The force in earth reference in x,y,z
                moment_earth : array-like
                    The moment in earth reference in x,y,z
                volume : float
                    The total volume displaced
                volume_centroid : array-like
                    The centroid of the volume displaced
                wetted_surface_area :float
                    The area of all faces below the waterplane
                waterplane_area : float
                    The area of the waterplane
                centre_of_flotation : array-like
                    The centroid of the waterplane
                Ax : float
                    The first moment of area in x of the waterplane
                Ay : float
                    The first moment of area in y of the waterplane
                Ax_origin : float
                    The first moment of area in x of the waterplane, taken about the waterplane origin
                Ay_origin : float
                    The first moment of area in y of the waterplane, taken about the waterplane origin
                Ixx : float
                    The second moment of area in x of the waterplane
                Iyy : float
                    The second moment of area in y of the waterplane
                Ixy : float
                    The second moment of area in both x and y of the waterplane
                Ixx_origin : float
                    The second moment of area in x of the waterplane, taken about the waterplane origin
                Iyy_origin : float
                    The second moment of area in y of the waterplane, taken about the waterplane origin
                Ixy_origin : float
                    The second moment of area in both x and y of the waterplane, taken about the waterplane origin
                Iu : float
                    The first principle second moment of area
                Iv : float
                    The second principle second moment of area
                theta_principle : float
                    The angle between the original and principle second moments of area, in degrees
                BMt : float
                    The transverse distance from the centre of buoyancy to the metacentre
                BMl : float
                    The longitudinal distance from the centre of buoyancy to the metacentre
                GMt : float
                    The transverse distance from the centre of gravity to the metacentre
                GMl : float
                    The longitudinal distance from the centre of gravity to the metacentre
                state : str
                    'Flying' if the boat is entirely above the water.
                    'Underwater' if the boat is entirely below the water.
                    'Floating' if the boat is partially immersed.
                Lwl : float
                    Length of waterline
                Bwl : float
                    Beam of waterline
                Cwp : float
                    The ratio of the waterplane area to Lwl * BWl
                bounds : array-like
                    A 3x2 array of min and max values, in the waterplanes x,y and z directions
                depth : float
                    The maximum depth of the boat below the waterplane
                """

                def __init__(self):
                    self.force = np.array([0.0, 0.0, 0.0])
                    self.moment = np.array([0.0, 0.0, 0.0])

                    self.force_earth = np.array([0.0, 0.0, 0.0])
                    self.moment_earth = np.array([0.0, 0.0, 0.0])

                    self.volume = 0.0
                    self.volume_centroid = np.array([0.0, 0.0, 0.0])

                    self.wetted_surface_area = 0.0

                    self.waterplane_area = 0.0
                    self.centre_of_flotation = np.array([0.0, 0.0, 0.0])

                    self.Ax = 0.0
                    self.Ay = 0.0
                    self.Ax_origin = 0.0
                    self.Ay_origin = 0.0

                    self.Ixx = 0.0
                    self.Iyy = 0.0
                    self.Ixy = 0.0
                    self.Ixx_origin = 0.0
                    self.Iyy_origin = 0.0
                    self.Ixy_origin = 0.0
                    self.Iu = 0.0
                    self.Iv = 0.0
                    self.theta_principle = 0.0

                    self.BMt = 0.0
                    self.BMl = 0.0
                    self.GMt = 0.0
                    self.GMl = 0.0
                    self.GZ_t = 0.0
                    self.GZ_l = 0.0

                    self.state = "Flying"  # Flying, Floating, Underwater

                    self.Lwl = 0.0
                    self.Bwl = 0.0
                    self.Cwp = 0.0

                    self.bounds = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
                    self.depth = 0.0

            class WeightResult:
                """Storage of individual weight results

                Attributes
                ----------
                force : array-like
                    The force in the body reference in x,y,z
                moment : array-like
                    The moment in body reference in x,y,z
                force_earth : array-like
                    The force in earth reference in x,y,z
                moment_earth : array-like
                    The moment in earth reference in x,y,z
                centre_of_gravity : array-like
                    A copy of the weights position, stored here for consistency and easy display
                """

                def __init__(self):
                    self.force = np.array([0.0, 0.0, 0.0])
                    self.moment = np.array([0.0, 0.0, 0.0])

                    self.force_earth = np.array([0.0, 0.0, 0.0])
                    self.moment_earth = np.array([0.0, 0.0, 0.0])

                    self.centre_of_gravity = np.array([0.0, 0.0, 0.0])

            def __init__(self):
                self.reset()

            def dict(self):
                return {
                    "volume_error": self.volume_error,
                    "heel": self.heel,
                    "trim": self.trim,
                    "fx": self.force[0],
                    "fy": self.force[1],
                    "fz": self.force[2],
                    "mx": self.moment[0],
                    "my": self.moment[1],
                    "mz": self.moment[2],
                    "fx_earth": self.force_earth[0],
                    "fy_earth": self.force_earth[1],
                    "fz_earth": self.force_earth[2],
                    "mx_earth": self.moment_earth[0],
                    "my_earth": self.moment_earth[1],
                    "mz_earth": self.moment_earth[2],
                    "fx_weight": self.weight_force[0],
                    "fy_weight": self.weight_force[1],
                    "fz_weight": self.weight_force[2],
                    "mx_weight": self.weight_moment[0],
                    "my_weight": self.weight_moment[1],
                    "mz_weight": self.weight_moment[2],
                    "cogx": self.centre_of_gravity[0],
                    "cogy": self.centre_of_gravity[1],
                    "cogz": self.centre_of_gravity[2],
                    "volume": self.volume,
                    "volume_centroid_x": self.volume_centroid[0],
                    "volume_centroid_y": self.volume_centroid[1],
                    "volume_centroid_z": self.volume_centroid[2],
                    "fx_buoyancy": self.buoyancy_force[0],
                    "fy_buoyancy": self.buoyancy_force[1],
                    "fz_buoyancy": self.buoyancy_force[2],
                    "mx_buoyancy": self.buoyancy_moment[0],
                    "my_buoyancy": self.buoyancy_moment[1],
                    "mz_buoyancy": self.buoyancy_moment[2],
                    "wetted_surface_area": self.wetted_surface_area,
                    "waterplane_area": self.waterplane_area,
                    "center_of_flotation_x": self.centre_of_flotation[0],
                    "center_of_flotation_y": self.centre_of_flotation[1],
                    "center_of_flotation_z": self.centre_of_flotation[2],
                    "Ax": self.Ax,
                    "Ay": self.Ay,
                    "Ax_origin": self.Ax_origin,
                    "Ay_origin": self.Ay_origin,
                    "Ixx": self.Ixx,
                    "Iyy": self.Iyy,
                    "Ixy": self.Ixy,
                    "Ixx_origin": self.Ixx_origin,
                    "Iyy_origin": self.Iyy_origin,
                    "Ixy_origin": self.Ixy_origin,
                    "Iu": self.Iu,
                    "Iv": self.Iv,
                    "theta_principle": self.theta_principle,
                    "BMt": self.BMt,
                    "BMl": self.BMl,
                    "GMt": self.GMt,
                    "GMl": self.GMl,
                    "state": self.state,
                    "Lwl": self.Lwl,
                    "Bwl": self.Bwl,
                    "Cwp": self.Cwp,
                    "depth": self.depth,
                    "GZ_l": self.GZ_l,
                    "GZ_t": self.GZ_t,
                }

            def reset(self):
                """Clears results and restores default values"""
                self.current = False
                self.partial = True
                self.heel = 0.0
                self.trim = 0.0

                self.force = np.array([0.0, 0.0, 0.0])
                self.moment = np.array([0.0, 0.0, 0.0])

                self.force_earth = np.array([0.0, 0.0, 0.0])
                self.moment_earth = np.array([0.0, 0.0, 0.0])

                self.weight_force = np.array([0.0, 0.0, 0.0])
                self.weight_moment = np.array([0.0, 0.0, 0.0])
                self.centre_of_gravity = np.array([0.0, 0.0, 0.0])

                self.volume = 0.0
                self.volume_centroid = np.array([0.0, 0.0, 0.0])
                self.buoyancy_force = np.array([0.0, 0.0, 0.0])
                self.buoyancy_moment = np.array([0.0, 0.0, 0.0])

                self.wetted_surface_area = 0.0

                self.waterplane_area = 0.0
                self.centre_of_flotation = np.array([0.0, 0.0, 0.0])

                self.Ax = 0.0
                self.Ay = 0.0
                self.Ax_origin = 0.0
                self.Ay_origin = 0.0

                self.Ixx = 0.0
                self.Iyy = 0.0
                self.Ixy = 0.0
                self.Ixx_origin = 0.0
                self.Iyy_origin = 0.0
                self.Ixy_origin = 0.0
                self.Iu = 0.0
                self.Iv = 0.0
                self.theta_principle = 0.0

                self.BMt = 0.0
                self.BMl = 0.0
                self.GMt = 0.0
                self.GMl = 0.0

                self.state = "Flying"  # Flying, Floating, Underwater

                self.Lwl = 0.0
                self.Bwl = 0.0
                self.Cwp = 0.0

                self.bounds = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
                self.depth = 0.0

                self.GZ_l = 0.0
                self.GZ_t = 0.0
                self.volume_error = 0.0

                self.mesh_results = {}
                self.weight_results = {}

            def __str__(self):
                """A plain text output of overall results

                Returns
                -------
                str
                    Plain text results
                """
                res = f"""
Forces:  
Buoyancy:   x = {self.buoyancy_force[0]:6.4e}, y = {self.buoyancy_force[1]:6.4e}, z = {self.buoyancy_force[2]:6.4e}  
Weight:     x = {self.weight_force[0]:6.4e}, y = {self.weight_force[1]:6.4e}, z = {self.weight_force[2]:6.4e}   
Total:      x = {self.force[0]:6.4e}, y = {self.force[1]:6.4e}, z = {self.force[2]:6.4e}   

Moments:  
Buoyancy    x = {self.buoyancy_moment[0]:6.4e}, y = {self.buoyancy_moment[1]:6.4e}, z = {self.buoyancy_moment[2]:6.4e}  
Weight      x = {self.weight_moment[0]:6.4e}, y = {self.weight_moment[1]:6.4e}, z = {self.weight_moment[2]:6.4e}   
Total:      x = {self.moment[0]:6.4e}, y = {self.moment[1]:6.4e}, z = {self.moment[2]:6.4e}  

Volume = {self.volume:6.4f}  
Center of Buoyancy:  
    x = {self.volume_centroid[0]:6.4f}, y = {self.volume_centroid[1]:6.4f}, z = {self.volume_centroid[2]:6.4f}  
Center of Gravity:  
    x = {self.centre_of_gravity[0]:6.4f}, y = {self.centre_of_gravity[1]:6.4f}, z = {self.centre_of_gravity[2]:6.4f}  
"""
                if not self.partial:
                    res += f"""
Wetted Surface Area = {self.wetted_surface_area:6.4f}

Waterplane Area = {self.waterplane_area:6.4f}
Center of Flotation:
    x = {self.centre_of_flotation[0]:6.4f}, y = {self.centre_of_flotation[1]:6.4f}, z = {self.centre_of_flotation[2]:6.4f}
Moment of Inertia:
    Ixx = {self.Ixx:6.4f}, Iyy = {self.Iyy:6.4f}, Ixy = {self.Ixy:6.4f}
    Iu = {self.Iu:6.4f}, Iv = {self.Iv:6.4f}, Theta = {self.theta_principle:6.4f}

Length Water Line = {self.Lwl:6.4f}
Beam Water Line = {self.Bwl:6.4f}
Waterplane Coefficient = {self.Cwp:6.4f}

BMt = {self.BMt:6.4f}, BMl = {self.BMl:6.4f}
GMt = {self.GMt:6.4f}, GMl = {self.GMl:6.4f}
"""
                return res

            def markdown(self):
                """A markdown output with tables for individual results and totals

                Returns
                -------
                str
                    markdown formatted results
                """
                return f"""
## Forces

| Name | Magitude | Position X | Position Y | Position Z | Moment X | Moment Y
|-|-|-|-|-|-|-|
|
| *Weights*
{'''
'''.join(f'| {name} | {res.force_earth[2]:.2e} N | {res.centre_of_gravity[0]:.2e} mm | {res.centre_of_gravity[1]:.2e} mm | {res.centre_of_gravity[2]:.2e} mm | {res.moment_earth[0]:.2e} N mm | {res.moment_earth[1]:.2e} N mm' for name,res in self.weight_results.items())}|
|
| *Meshes*
{'''
'''.join(f'| {name} | {res.force_earth[2]:.2e} N | {res.volume_centroid[0]:.2e} mm | {res.volume_centroid[1]:.2e} mm | {res.volume_centroid[2]:.2e} mm | {res.moment_earth[0]:.2e} N mm | {res.moment_earth[1]:.2e} N mm ' for name,res in self.mesh_results.items())}|
| 
| **Total** | {self.force_earth[2]:.2e} N | | | | {self.moment_earth[0]:.2e} N mm | {self.moment_earth[1]:.2e} N mm |

## Immersion Properties

| Name | Volume Displaced | Depth | Wetted Surface Area | State 
|-|-|-|-|-|
{'''
'''.join(f'| {name} | {res.volume:.2e} mm^3 | {res.depth:.2e} mm | {res.wetted_surface_area:.2e} mm^2 | {res.state} |' for name,res in self.mesh_results.items())}|
|
| **Total** | {self.volume:.2e} mm^3 | {self.depth:.2e} mm | {self.wetted_surface_area:.2e} | {self.state} |

## Waterplane Area Properties

| Name | Waterplane Area | Centre of Flotation X | Centre of Flotation Y | Centre of Flotation Z | Lwl | Bwl | Cwp | 
|-|-|-|-|-|-|-|-|
{'''
'''.join(f'| {name} | {res.waterplane_area:.2e} mm^2 | {res.centre_of_flotation[0]:.2e} mm |{res.centre_of_flotation[1]:.2e} mm | {res.centre_of_flotation[2]:.2e} mm | {res.Lwl:.2e} mm | {res.Bwl:.2e} mm | {res.Cwp:.2f} |' for name,res in self.mesh_results.items())}|
|
| **Total** | {self.waterplane_area:.2e} mm^2 | {self.centre_of_flotation[0]:.2e} mm | {self.centre_of_flotation[1]:.2e} mm | {self.centre_of_flotation[2]:.2e} mm | {self.Lwl:.2e} mm | {self.Bwl:.2e} mm | {self.Cwp:.2f} |

## Waterplane Moment of Inertia

| Name | Ixx | Iyy | Ixy | Iu | Iv | θ |
|-|-|-|-|-|-|-|
{'''
'''.join(f'| {name} | {res.Ixx:.2e} mm^4 | {res.Iyy:.2e} mm^4 | {res.Ixy:.2e} mm^4 | {res.Iu:.2e} mm^4 | {res.Iv:.2e} mm^4 | {res.theta_principle:.2e}°' for name,res in self.mesh_results.items())}|
|
| **Total** | {self.Ixx:.2e} mm^4 | {self.Iyy:.2e} mm^4 | {self.Ixy:.2e} mm^4 | {self.Iu:.2e} mm^4 | {self.Iv:.2e} mm^4 | {self.theta_principle:.2e}°|

## Metacentre Distances

| Name | BMt | BMl | GMt | GMl | GZ |
|-|-|-|-|-|-|
{'''
'''.join(f'| {name} | {res.BMt:.2e} mm | {res.BMl:.2e} mm | {res.GMt:.2e} mm | {res.GMl:.2e} mm | {res.GZ_t:.2e}' for name,res in self.mesh_results.items())}|
|
| **Total** | {self.BMt:.2e} mm | {self.BMl:.2e} mm | {self.GMt:.2e} mm | {self.GMl:.2e} mm | {self.GZ_t:.2e}
"""

    if "pyvista" in sys.modules:  # Display

        def plot_below_surface(self, reference="body"):
            """Plotting mesh faces below the waterplane

            Yields
            ------
            pv.PolyData
                A pyvista mesh for plotting
            """
            for name, mesh in self.faces_below_water.items():
                if (
                    self.active_mesh[name]
                    and self.show_mesh[name]
                    and len(mesh.faces) > 0
                ):
                    m = mesh.visualise()
                    if reference == "body":
                        yield m
                    elif reference == "earth":
                        m.rotate_x(self.heel)
                        m.rotate_y(self.trim)
                        m.rotate_z(self.leeway)
                        yield m

        def plot_transformed(self, reference="body"):
            """Plots the full meshes at the current transformations

            Yields
            ------
            pv.PolyData
                A pyvista mesh for plotting
            """
            for name, mesh in self.transformed.items():
                if self.show_mesh[name]:
                    m = mesh.visualise()
                    if reference == "body":
                        yield m
                    elif reference == "earth":
                        m.rotate_x(self.heel)
                        m.rotate_y(self.trim)
                        m.rotate_z(self.leeway)
                        yield m

        def plot_weights(self, reference="body"):
            """Generates a representation of the weights

            The weight size is set so the volume of the sphere is the same as the volume
            needed to generate enough bouyancy force to be in static equilibrium.

            Yields
            ------
            Pv.Sphere
                A pyvista sphere at the location of the weight
            """
            for name, weight in self.weight_forces.items():
                if self.show_weight[name]:
                    s = pv.Sphere(
                        center=weight[0],
                        radius=(
                            (3 * weight[1]) / (4 * self.water_density * self.g * np.pi)
                        )
                        ** (1 / 3),
                    )
                    if reference == "body":
                        yield s
                    elif reference == "earth":
                        s.rotate_x(self.heel)
                        s.rotate_y(self.trim)
                        s.rotate_z(self.leeway)
                        yield s

        def plot_water_plane(self, reference="body"):
            """Plots the waterplane

            Returns
            -------
            pv.Plane
                The waterplane
            """
            centre = (
                self.results.centre_of_flotation
                if self.results.waterplane_area > 1e-8
                else self.waterplane_origin
            )
            if reference == "body":
                plane = pv.Plane(
                    centre,
                    self.R @ np.array([0, 0, 1]),
                    (self.results.bounds[0, 1] - self.results.bounds[0, 0]) * 2,
                    (self.results.bounds[1, 0] - self.results.bounds[1, 1]) * 2,
                )
            elif reference == "earth":
                plane = pv.Plane(
                    np.linalg.solve(self.R, centre),
                    np.array([0, 0, 1]),
                    (self.results.bounds[0, 1] - self.results.bounds[0, 0]) * 2,
                    (self.results.bounds[1, 0] - self.results.bounds[1, 1]) * 2,
                )
            else:
                plane = None
            return plane

        def plot_bounding_box(self, reference="body"):
            """Plots a bounding box around all meshes oriented with the waterplane

            Returns
            -------
            pv.Cube
                The bounding box
            """
            b = self.bounds()

            c = pv.Cube(bounds=(b[0, 0], b[0, 1], b[1, 0], b[1, 1], b[2, 0], b[2, 1]))
            if reference == "body":
                c.rotate_z(-self.leeway)
                c.rotate_y(-self.trim)
                c.rotate_x(-self.heel)
            return c

        def plot_centres(self, reference="body"):
            c = pv.PolyData(
                np.array(
                    [
                        self.results.centre_of_gravity,
                        self.results.centre_of_flotation,
                        self.results.volume_centroid,
                    ]
                )
            )
            if reference == "earth":
                c.rotate_x(self.heel)
                c.rotate_y(self.trim)
                c.rotate_z(self.leeway)
            c["Labels"] = ["CoG", "CoF", "CoB"]
            return c

    if True:  # Renaming

        def rename_mesh(self, old, new):
            """Helper function to rename the mesh properly

            Updates all relevant objects to new name

            Note
            ----
            If the old name does not exist no change is made.
            """
            try:
                self.meshes[new] = self.meshes.pop(old)
                self.transformed[new] = self.transformed.pop(old)
                self.active_mesh[new] = self.active_mesh.pop(old)
                self.show_mesh[new] = self.show_mesh.pop(old)
                self.local_position[new] = self.local_position.pop(old)
                self.local_rotation[new] = self.local_rotation.pop(old)
                self.faces_below_water[new] = self.faces_below_water.pop(old)
                self.water_edge[new] = self.water_edge.pop(old)
                self.run()
            except KeyError:
                pass
                # print("Name doesn't exist")

        def rename_weight(self, old, new):
            """Helper function to rename the weight properly

            Updates all relevant objects to new name

            Note
            ----
            If the old name does not exist no change is made.
            """
            try:
                self.weight_forces[new] = self.weight_forces.pop(old)
                self.active_weight[new] = self.active_weight.pop(old)
                self.show_weight[new] = self.show_weight.pop(old)
                self.run()
            except KeyError:
                pass
                # print("Name doesn't exist")


def load_hydro(filename):
    """Simple wrapper over pickle to load files"""
    with open(filename, "rb") as f:
        res = pickle.load(f)
    return res


'''
if __name__ == "__main__":

    h = BuoyancyModel()
    
    h.load_mesh("./data/hydrostatics_data/F50HullSTB.dxf")
    h.meshes['F50HullSTB.dxf_0'] = close_ends(mirror_uv(h.meshes['F50HullSTB.dxf_0'])) # Without closing faces, produces correct area

    h.set_weight_force("hull", np.array([0,0,0]), 2979)
    """
    h.meshes["cube"] = close_ends(mirror_uv(np.flip(np.array([
        [
            [1000,0,1000],
            [-1000,0,1000],
            [-1000,0,1000]
        ],
        [
            [1000,1000,1000],
            [-1000,1000,1000],
            [-1000,0,1000]
        ],
        [
            [1000,1000,-1000],
            [-1000,1000,-1000],
            [-1000,0,-1000]
        ],
        [
            [1000,0,-1000],
            [-1000,0,-1000],
            [-1000,0,-1000]
        ]
    ]),0)))
    h.active_mesh["cube"] = True
    h.local_position["cube"] = np.array([0,0,0])
    h.local_rotation["cube"] = np.array([0,0,0])
    """
    h.calculate_transformation()

    print(h.run())
    start = time()
    for i in range(1):
        h.heel = 0
        h.trim = 0
        h.height = i
        f,m = h.run()
    print(f, m)
    print(time()-start)
    h.calculate_results()
    print("done")
    """    
    p = pv.Plotter()
    for mesh in h.plot_below_surface():
        p.add_mesh(mesh)
    p.show()
    """
    
    
'''

if __name__ == "__main__":
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

    assert abs(b.results.volume - 1195654202.10127) / b.results.volume < 1e-8
    assert (
        abs(b.results.bounds[2, 0] + 265.438659667968) / b.results.bounds[2, 0] < 1e-8
    )
    assert abs(b.results.Lwl - 13659.6181685288) / b.results.Lwl < 1e-8
    assert abs(b.results.Bwl - 690.086370260558) / b.results.Bwl < 1e-8
    assert (
        abs(b.results.wetted_surface_area - 10136608.4581886)
        / b.results.wetted_surface_area
        < 1e-8
    )
    assert (
        abs(b.results.waterplane_area - 7752889.78609542) / b.results.waterplane_area
        < 1e-8
    )
    assert abs(b.results.Cwp - 0.822472906915916) / b.results.Cwp < 1e-8
    assert abs(b.results.BMt - 0.197019473752493 * 1000) / b.results.BMt < 1e-8
    assert abs(b.results.BMl - 81.7520382810099 * 1000) / b.results.BMl < 1e-8
    assert abs(b.results.GMt + 2.14409903627079 * 1000) / b.results.GMt < 1e-8
    assert abs(b.results.GMl - 79.4109197709866 * 1000) / b.results.GMl < 1e-6
    assert abs(b.results.Ixx - 235567161687.948) / b.results.Ixx < 1e-8
    assert abs(b.results.Iyy - 97747168101033) / b.results.Iyy < 1e-8
    assert abs(b.results.Ixy - 475400280.56102) / b.results.Ixy < 1e-6
    assert abs(b.results.Iu - 97747168103350.7) / b.results.Iu < 1e-8
    assert abs(b.results.Iv - 235567159370.216) / b.results.Iv < 1e-8
    assert (
        abs(b.results.theta_principle - 45.0002793352728) / b.results.theta_principle
        < 1e-4
    )
    assert np.all(
        np.isclose(
            b.results.volume_centroid,
            np.array([6193.01396622061, -3737.52760421445, -93.6758117708258]),
        )
    )
    assert np.all(
        np.isclose(
            b.results.centre_of_flotation[0:2],
            np.array([6564.94484608725, -3737.46471134494]),
        )
    )
