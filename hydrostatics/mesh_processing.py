import sys
import numpy as np
import ezdxf as ez
import trimesh

try:
    import pyvista as pv
except:
    pass

from hydrostatics.fast_calcs.volume_calcs import volume_properties
from hydrostatics.fast_calcs.clipping_calcs import clip_points

if True:  # UV processing

    def read_data(filename: str):
        """Read DXF file data for a mesh

        This method is tailored to find the vertices of an AcDbPolyFaceMesh object. It utilises ezdxf for parsing.
        As an artifact, it produces many points at the origin, which we manually remove.

        Parameters
        ----------
        filename : str
            path to the .dxf file

        Returns
        -------
        list
            A list of 'meshes'. Each mesh is a Nx3 array of vertices in 3D.

        Warnings
        --------
        All points exactly at the origin will be removed
        """

        meshes = []

        doc = ez.readfile(filename)
        msp = doc.modelspace()  # Should list all the entities in the file
        for entity in msp:
            if (
                entity.get_mode() == "AcDbPolyFaceMesh"
            ):  # Surface meshes seem to always have this type. Other entities may work, but not sure
                v = np.array(
                    [
                        [p.x, p.y, p.z]
                        for p in entity.points()
                        if p.x != 0.0 and p.y != 0.0 and p.z != 0.0
                    ]
                )  # Get all points and remove [0,0,0] points. Seems to find lots of these
                meshes.append(v)

        return meshes

    # TODO: Inversion
    def convert_to_uv(mesh: np.array, shape: tuple):
        """Converts an array of vertices to a 2D UV mesh of vertices

        Parameters
        ----------
        mesh : np.array
            A Nx3 array of vertices in 3D
        shape : (int, int)
            The number of rows and columns in the UV mesh

        Returns
        -------
        np.array
            NxMx3. The first two dimensions are U and V, and
            the third dimension is values in x, y and z.
        """
        uv = np.reshape(mesh, (*shape, 3), "F")

        if False:
            # Invert
            uv = np.flip(uv, 0)

        return uv

    def mirror_uv(uv: np.array):
        """Mirrors half to create full mesh

        Parameters
        ----------
        uv : np.array
            A NxMx3 UV mesh

        Returns
        -------
        np.array
            The original mesh plus it's mirror, if the UV is not closed. Otherwise, the same as the input.
        """
        if not isinstance(uv, np.ndarray):
            return uv

        if not top_closed(uv):
            m = np.flip(np.copy(uv), 0)
            m[:, :, 1] = m[:, :, 1] - 2 * (
                m[:, :, 1] - m[-1, :, 1]
            )  # Mirror y axis about the centerline

            return np.flip(
                np.concatenate((m, uv[1:, :, :]), 0), 0
            )  # Not sure why the whole thing is flipped again, but needed for output to be the same as spreadsheet. Might be due to inversion
        return uv

    def close_ends(uv: np.array):
        """Close off open faces of mesh

        Checks if left and right edges are symmetric. If not, adds new columns at the average y position.
        Checks if the top is equal to the bottom. If not, adds new rows at the average y position.

        Nees testing: this probably doesn't cover all cases

        Parameters
        ----------
        uv : np.array
            NxMx3 UV mesh

        Returns
        -------
        np.array
            a closed version of the input
        """
        if not isinstance(uv, np.ndarray):
            return uv

        y_plane = np.mean(uv[0, :, 1])

        if not left_closed(uv):
            left = np.copy(uv[:, 0, :])
            left[:, 1] = y_plane
            uv = np.insert(uv, 0, left, axis=1)
        if not right_closed(uv):
            right = np.copy(uv[:, -1, :])
            right[:, 1] = y_plane
            uv = np.insert(uv, -1, right, axis=1)
        if not top_closed(uv):
            top = np.copy(uv[0, :, :])
            top[:, 1] = y_plane
            uv = np.insert(uv, 0, top, axis=0)
            uv = np.insert(uv, -1, top, axis=0)

        return uv

    def top_closed(uv: np.array):
        """Checks if the top row is approximately equal to the bottom row

        Parameters
        ----------
        uv : np.array
            NxMx3 UV mesh

        Returns
        -------
        bool
            True if the top is closed
        """
        return np.all(abs(uv[0, :, :] - uv[-1, :, :]) < 1e-5)

    def left_closed(uv: np.array):
        """Checks if the left column is symmetric

        Parameters
        ----------
        uv : np.array
            NxMx3 UV mesh

        Returns
        -------
        bool
            True if the left is closed
        """
        y_plane = np.mean(uv[:, 0, 1])
        return all(abs(uv[:, 0, 1] - y_plane) < 1e-5)

    def right_closed(uv: np.array):
        """Checks if the right column is symmetric

        Parameters
        ----------
        uv : np.array
            NxMx3 UV mesh

        Returns
        -------
        bool
            True if the right is closed
        """
        y_plane = np.mean(uv[:, -1, 1])
        return all(abs(uv[:, -1, 1] - y_plane) < 1e-5)

    def print_to_csv(uv: np.array, filename: str):
        """Writes a 3D array out in a format comparable to the original excel spreadsheet

        3D array is flattened to 2D by inserting a number of columns next to each existing column equal to the size of the third dimension.

        Parameters
        ----------
        uv : np.array
            NxMx3 UV mesh
        filename : str
            The file to write the csv to
        """
        with open(filename, "w") as f:
            f.writelines(
                "\n".join(
                    ",".join(
                        ",".join(f"{uv[i,j,k]:.6g}" for k in range(uv.shape[2]))
                        for j in range(uv.shape[1])
                    )
                    for i in range(uv.shape[0])
                )
            )

    if "pyvista" in sys.modules:

        def plot_uv(uv: np.array):
            """Creates a polymesh from a uv mesh data for use in pyvista

            Parameters
            ----------
            uv : np.array
                NxMx3 UV mesh

            Returns
            -------
            pv.PolyData
                A pyvista mesh object
            """
            vertices = np.reshape(uv, (-1, 3), "F")
            faces = []
            for i in range(uv.shape[0] - 1):
                for j in range(uv.shape[1] - 1):
                    faces.append(4)
                    faces.extend(
                        np.ravel_multi_index(
                            ((i, i + 1, i + 1, i), (j, j, j + 1, j + 1)),
                            uv.shape[0:2],
                            order="F",
                        )
                    )
            return pv.PolyData(vertices, np.array(faces))

        def plot_trimesh(mesh: trimesh.Trimesh):
            """Creates a polymesh from a trimesh object

            Parameters
            ----------
            trimesh.Trimesh
                A 3D trimesh object

            Returns
            -------
            pv.PolyData
                A pyvista mesh object
            """
            return pv.PolyData(
                mesh.vertices,
                np.concatenate(
                    (3 * np.ones((mesh.faces.shape[0], 1)), mesh.faces), axis=1
                ),
            )

    def save_uv(uv: np.array, filename):
        """Saves a uv mesh to a file
        Probably unnecessary, essentially a copy from numpy
        """
        np.save(filename, uv)

    def load_uv(filename):
        """Loads a uv mesh from a file
        Probably unnecessary, essentially a copy from numpy
        """
        return np.load(filename)

    def uv_to_trimesh(uv: np.array):
        """Converts a UV mesh to a trimesh

        Parameters
        ----------
        uv : np.array
            NxMx3 UV mesh

        Returns
        -------
        trimesh.Trimesh
        """
        i0 = np.array([i for i in range(uv.shape[0] - 1)] * (uv.shape[1] - 1))
        i1 = i0 + 1
        j0 = np.reshape(
            [[i] * (uv.shape[0] - 1) for i in range(uv.shape[1] - 1)], (1, -1)
        )[0]
        j1 = j0 + 1

        tl = np.ravel_multi_index((i0, j0), uv.shape[0:2], order="F")
        bl = np.ravel_multi_index((i1, j0), uv.shape[0:2], order="F")
        tr = np.ravel_multi_index((i0, j1), uv.shape[0:2], order="F")
        br = np.ravel_multi_index((i1, j1), uv.shape[0:2], order="F")

        a = np.array(
            [
                [tl[i], bl[i], tr[i]]
                for i in range((uv.shape[0] - 1) * (uv.shape[1] - 1))
            ]
        )
        b = np.array(
            [
                [tr[i], bl[i], br[i]]
                for i in range((uv.shape[0] - 1) * (uv.shape[1] - 1))
            ]
        )

        return trimesh.Trimesh(
            vertices=np.reshape(uv, (-1, 3), order="F"), faces=np.vstack((a, b))
        )


if True:  # Clipping and Culling
    '''
    #@profile
    def clip_points(vertices, faces, position=np.array([0.,0.,0.]), normal=np.array([0.,0.,1.])):
        """ Clips faces to below a plane

        This takes inspiration from the original, http://paulbourke.net/geometry/polygonmesh/source3.c.
        However, it replaces as many operations with numpy functions as possible.
        The code is not pretty, but the performance is around 100x that of naive python code in C style.
        This also makes up the bulk of the buoyancy force calculation, so it makes sense to optimise as much as possible.

        Parameters
        ----------
        vertices : np.array
            Nx3 array of vertices in 3D
        faces : np.array
            Mx3 array of vertex indices into `vertices`. The three points
            describe a triangular face.
        position : np.array
            A point on the plane in 3D
        normal : np.array
            The normal direction of the plane

        Returns
        -------
        vertices : np.array
            Nx3 array of vertices in 3D
        edges : np.array
            Nx2 array of vertex indices into `vertices`. The two points descibe one edge.
        faces : np.array
            Nx3 array of vertex indices into `vertices`. The three points
            describe a triangular face.
        """
        v_n = vertices @ normal
        p_n = position @ normal

        below = v_n[faces] <= p_n # Find truth table of points below the plane
        below_sum = below.sum(axis=1)
        #below_zero = below_sum==0
        below_one = below_sum==1
        below_two = below_sum==2
        below_three = below_sum==3

        faces_below = faces[below_three] # Get faces with all points below
        faces_one = faces[below_one] # Get faces with one points below

        # Find which vertices are above and which is below for faces with one point below
        vert_above = vertices[faces_one[np.logical_not(below[below_one])].reshape(-1,2)]
        vert_below = vertices[faces_one[below[below_one]]]

        # Replace indices above with new unused indices
        replacement_1 = np.vstack((vertices.shape[0] + np.arange(faces_one.shape[0]), (vertices.shape[0] + faces_one.shape[0] + np.arange(faces_one.shape[0]))))
        np.place(faces_one, np.logical_not(below[below_one]), replacement_1.flatten(order='F'))

        # Compute plane intersections
        a = vert_below - (vert_below@normal[:,None]-position@normal) * (vert_above[:,0] - vert_below) / (vert_above[:,0]@normal - vert_below@normal)[:,None]
        b = vert_below - (vert_below@normal[:,None]-position@normal) * (vert_above[:,1] - vert_below) / (vert_above[:,1]@normal - vert_below@normal)[:,None]

        # Append new vertices
        vertices = np.vstack((vertices, a, b))

        # Find which vertices are above and which is below for faces with two points below
        faces_two = faces[below_two]
        vert_above = vertices[faces_two[np.logical_not(below[below_two])]]
        vert_below = vertices[faces_two[below[below_two]].reshape(-1,2)]

        # Get new vertex indices
        replacement_2 = np.vstack((vertices.shape[0] + np.arange(faces_two.shape[0]), (vertices.shape[0] + faces_two.shape[0] + np.arange(faces_two.shape[0]))))
        replacement_2 = np.where(below[below_two][:,1], replacement_2, np.flipud(replacement_2)).T

        # Partition faces into two new faces
        faces_two_below = below[below_two]
        faces_two_a = np.where(faces_two_below[:,1] ,faces_two[faces_two_below].reshape(-1,2).T, np.flipud(faces_two[faces_two_below].reshape(-1,2).T)).T
        faces_two_b = faces_two_a[:,0] # pylint: disable=unsubscriptable-object

        # Add new vertex indices
        faces_two_a = np.hstack((faces_two_a, replacement_2[:,1][:,None])) # pylint: disable=unsubscriptable-object
        faces_two_b = np.hstack((replacement_2, faces_two_b[:,None]))

        # Compute plane intersections
        a = vert_above - (vert_above@normal[:,None]-position@normal) * (vert_below[:,0] - vert_above) / (vert_below[:,0]@normal - vert_above@normal)[:,None]
        b = vert_above - (vert_above@normal[:,None]-position@normal) * (vert_below[:,1] - vert_above) / (vert_below[:,1]@normal - vert_above@normal)[:,None]

        # Transform old replacements for use as edges
        replacement_1 = np.where(below[below_one][:,1], np.flipud(replacement_1), replacement_1).T

        # Collect final values in correct formats
        vertices = np.vstack((vertices, a, b))
        faces = np.vstack((faces_below, faces_one, faces_two_a, np.fliplr(faces_two_b)))
        edges = np.fliplr(np.vstack((replacement_1, np.fliplr(replacement_2))))

        return vertices, edges, faces
    '''

    def cull_interior(faces, edges, vertices, normal=np.array([0.0, 0.0, 1.0])):
        """Removes interior volumes split by waterplane

        For buoyancy calculations, only the volume displaced matters.
        Any interior volumes are not filled with water.
        Assuming we have cut a mesh in two, we have a set of edges which we have cut along.
        We partition these edges into loops and only keep the outermost loops.
        We also remove all faces attached to removed edges.
        This method should work for any mesh topologically equivalent to a 2-sphere (i.e. a 3D sphere)

        Note that the topology must still be a continuous outer surface. Interior bubbles will
        not be detected.

        Parameters
        ----------
        faces : np.array
            Nx3 array of vertex indices into `vertices`. The three points
            describe a triangular face.
        edges : np.array
            Nx2 array of vertex indices into `vertices`. The two points descibe one edge.
        vertices : np.array
            Nx3 array of vertices in 3D
        normal : np.array
            The direction of the cutting plane previously used, in 3D

        Returns
        -------
        faces : np.array
            Nx3 array of vertex indices into `vertices`. The three points
            describe a triangular face.
        edges : np.array
            Nx2 array of vertex indices into `vertices`. The two points descibe one edge.

        See Also
        --------
        cull_interior_edges, cull_interior_faces
        """
        edges, interior_points = cull_interior_edges(edges, vertices, normal)
        faces = cull_interior_faces(faces, interior_points)
        return faces, edges

    def cull_interior_faces(faces, interior_points):
        """Remove faces connected to points

        Removes all faces connected to points

        Parameters
        ----------
        faces : np.array
            Nx3 array of vertex indices into `vertices`. The three points
            describe a triangular face.
        interior_points : np.array
            1D array of indices into `vertices` to remove.

        Returns
        -------
        faces : np.array
            Nx3 array of vertex indices into `vertices`. The three points
            describe a triangular face.

        See Also
        --------
        cull_interior
        """
        removed = True
        new = np.isin(faces, interior_points).any(axis=1)
        interior_faces = np.vstack((new, new, new)).T
        s = interior_faces.sum()
        while removed:
            removed = False
            new = np.isin(faces, faces[interior_faces]).any(axis=1)
            interior_faces = np.vstack((new, new, new)).T
            if s != interior_faces.sum():
                removed = True
                s = interior_faces.sum()

        return faces[np.logical_not(interior_faces[:, 0]), :]

    def cull_interior_edges(edges, vertices, normal=np.array([0.0, 0.0, 1.0])):
        """Removes edges inside loops of edges

        Assumes geometry is well behaved, so loops do not cross

        Parameters
        ----------
        edges : np.array
            Nx2 array of vertex indices into `vertices`. The two points descibe one edge.
        vertices : np.array
            Nx3 array of vertices in 3D
        normal : np.array
            The direction of the cutting plane previously used, in 3D

        Returns
        -------
        interior_points : np.array
            1D array of indices into `vertices` that have been removed

        See Also
        --------
        cull_interior
        """
        x = np.cross(normal, np.random.rand(3))
        y = np.cross(normal, x)

        # undecided = np.full(edges.shape, True)
        loops = []
        """
        while np.any(undecided):
            loop = np.full(edges.shape, False)
            loop[np.argmax(undecided[:,0]),:] = True
            added  = True
            s = loop.sum()
            while added:
                added = False
                new = np.isin(edges, edges[loop]).any(axis=1)
                loop = np.vstack((new,new)).T
                if loop.sum() != s:
                    added = True
                    s = loop.sum()

            np.copyto(undecided, np.logical_not(loop), where=loop)
            loops.append(edges[loop[:,0],:])
            ----------------------------
        A = np.zeros((vertices.shape[0],vertices.shape[0]), dtype=int)
        A[edges[:,0],edges[:,1]] = 1        
        G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
        for c in nx.strongly_connected_components(G):
            if len(c) > 1:
                loops.append(np.array(G.subgraph(c).edges))
        """

        tail_index = {v: i for i, v in enumerate(edges[:, 0])}
        visited = np.full(edges.shape[0], False)
        while not np.all(visited):
            new = np.argmin(visited)
            loop = []
            start = edges[new, 0]
            visited[new] = True
            loop.append(new)
            cycle = False
            finished = False
            while not finished:
                head = edges[new, 1]
                if head in tail_index:
                    new = tail_index[head]
                    if not visited[new]:
                        visited[new] = True
                        loop.append(new)
                    elif edges[new, 0] == start:
                        finished = True
                        cycle = True
                    else:
                        finished = True
                else:
                    finished = True
            if cycle:
                loops.append(edges[loop, :])

        interior_loops = np.array([False for loop in loops])
        for i, loop_1 in enumerate(loops):
            for j, loop_2 in enumerate(loops):
                if (
                    i != j
                    and not interior_loops[j]
                    and wn_PnPoly(loop_1[0, 0], loop_2, vertices, x, y)
                ):  # Check if loop is inside another loop
                    interior_loops[i] = True
                    break
            if not interior_loops[i]:  # Remove loop if not closed properly
                _, counts = np.unique(loop_1, return_counts=True)
                if (counts == 1).any():
                    interior_loops[i] = True

        exterior = [loop for i, loop in enumerate(loops) if not interior_loops[i]]
        interior = [loop for i, loop in enumerate(loops) if interior_loops[i]]
        if len(exterior) > 0:
            exterior = np.vstack(exterior)
        else:
            exterior = np.array([])

        if len(interior) > 0:
            interior = np.vstack(
                [loop for i, loop in enumerate(loops) if interior_loops[i]]
            ).flatten()
        else:
            interior = np.array([])

        return exterior, interior

    def wn_PnPoly(p, poly, vertices, x, y):
        """Chekk if a point is inside a polygon

        Algorithm adapted from http://geomalgorithms.com/a03-_inclusion.html

        Parameters
        ----------
        p : int
            index of the point to check in vertices
        poly : np.array
            Nx2 array of vertex indices into `vertices`. The two points descibe one edge.
        vertices : np.array
            Nx3 array of vertices in 3D
        x : np.array
            Unit vector in the x direction
        y : np.array
            Unit vector in th y direction

        Returns
        -------
        int
            The number of times the point is found to be inside the polygon.
        """
        n = 0
        for edge in poly:
            if vertices[edge[0]] @ y <= vertices[p] @ y:
                if vertices[edge[1]] @ y > vertices[p] @ y:
                    if (
                        is_left(vertices[edge[0]], vertices[edge[1]], vertices[p], x, y)
                        > 0
                    ):
                        n += 1
            else:
                if vertices[edge[1]] @ y <= vertices[p] @ y:
                    if (
                        is_left(vertices[edge[0]], vertices[edge[1]], vertices[p], x, y)
                        < 0
                    ):
                        n -= 1
        return n

    def is_left(a, b, c, x, y):
        """Helper for wn_PnPoly"""
        return (b @ x - a @ x) * (c @ y - a @ y) - (c @ x - a @ x) * (b @ y - a @ y)


if True:  # Mesh Classes

    class Mesh:
        """Stores mesh data in a general format

        Attributes
        ----------
        vertices : np.array
            Nx3 array of vertices in 3D
        faces : np.array
            Mx3 array of vertex indices into `vertices`. The three points
            describe a triangular face.
        """

        def __init__(self, vertices, faces):
            self.vertices = vertices
            self.faces = faces

        def cut(
            self,
            position=np.array([0.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            include_interior=False,
        ):
            """Cuts the mesh by a plane

            Parameters
            ----------
            position : np.array
                A point on the plane in 3D
            normal : np.array
                The normal direction of the plane
            include_interior : bool
                Whether to keep interior faces in new mesh and slice

            Returns
            -------
            m : Mesh
                A new mesh containing faces below the plane
            s : Slice
                A slice of edges where the mesh intersects the plane
            """
            vertices, edges, faces = clip_points(
                self.vertices, self.faces, position=position, normal=normal
            )

            if not include_interior:
                vertices, index = np.unique(
                    vertices.round(decimals=8), return_inverse=True, axis=0
                )

                faces = index.astype(np.int32)[faces]
                edges = index.astype(np.int32)[edges]
                edges = edges[np.where(edges[:, 0] != edges[:, 1])]  # Remove null edges
                faces, edges = cull_interior(faces, edges, vertices, normal=normal)

            m = Mesh(vertices, faces)
            s = Slice(vertices, edges)
            s.centre = position
            return m, s

        def volume_properties(self, centre=np.array([0.0, 0.0, 0.0])):
            """Calculates volume and centroid

            Parameters
            ----------
            centre : np.array
                The volumes are calculated as tetrahedrons. The centre specifies the fourth point of each tetrahedron

            Returns
            -------
            volume : float
                The total volume
            centroid : np.array
                The centroid of the volume
            """
            """
            if len(self.faces) == 0:
                return 0, np.array([0.,0.,0.])
            volumes = np.linalg.det(self.vertices[self.faces] - centre)/6.0 #np.array([np.cross(self.vertices[f[0]] - centre, self.vertices[f[1]] - centre).dot(self.vertices[f[2]] - centre) / 6.0 for f in self.faces])
            centroids = (self.vertices[self.faces].sum(axis=1)+centre)/4.0 #np.array([(self.vertices[f[0]] + self.vertices[f[1]] + self.vertices[f[2]] + centre)/4.0 for f in self.faces])
            volume = sum(volumes)
            if volume > 0:
                centroid = volumes@centroids / volume
            else:
                centroid = np.array([0.,0.,0.])
            return volume, centroid
            """
            if self.vertices.shape[0] > 0 and self.faces.shape[0] > 2:
                return volume_properties(self.vertices, self.faces, centre)
            return 0.0, np.array([0.0, 0.0, 0.0])

        def area(self):
            """Calculates the outer surface area of the mesh

            Returns
            -------
            float
                The surface area
            """
            if len(self.faces) == 0:
                return 0
            return sum(
                np.linalg.norm(
                    np.cross(
                        self.vertices[self.faces][:, 1, :]
                        - self.vertices[self.faces][:, 0, :],
                        self.vertices[self.faces][:, 2, :]
                        - self.vertices[self.faces][:, 0, :],
                    ),
                    axis=1,
                )
                / 2
            )  # sum(0.5 * np.linalg.norm(np.cross(self.vertices[f[1]]-self.vertices[f[0]], self.vertices[f[2]]-self.vertices[f[0]])) for f in self.faces)

        if "pyvista" in sys.modules:

            def visualise(self):
                """Converts to pyvista mesh for plotting

                Returns
                -------
                pv.PolyData
                    A pyvista mesh
                """
                return pv.PolyData(
                    self.vertices.copy(),
                    np.concatenate(
                        (
                            3 * np.ones((self.faces.shape[0], 1), dtype=int),
                            self.faces,
                        ),
                        axis=1,
                    ),
                )

        def save(self, filename, rotation=(0, 0, 0)):
            """Saves the mesh to a file

            Converts the mesh to trimesh format and runs the export method

            Parameters
            ----------
            filename : str
                The file to save to
            """
            t = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
            t.apply_transform(
                trimesh.transformations.euler_matrix(
                    np.deg2rad(rotation[0]),
                    np.deg2rad(rotation[1]),
                    np.deg2rad(rotation[2]),
                    axes="sxyz",
                )
            )
            t.export(filename)

    class Slice:
        """A set of edges produced from the intersection of a mesh and a plane

        Attributes
        ----------
        vertices : np.array
            Nx3 array of vertices in 3D
        edges : np.array
            Nx2 array of vertex indices into `vertices`. The two points descibe one edge.
        x : np.array
            A unit vector in the x direction
        y : np.array
            A unit vector in the y direction
        centre : np.array
            A centre-point to calculate moments about
        """

        def __init__(
            self,
            vertices,
            edges,
        ):
            self.vertices = vertices
            self.edges = edges
            self.x = np.array([1.0, 0.0, 0.0])
            self.y = np.array([0.0, 1.0, 0.0])
            self.centre = np.array([0.0, 0.0, 0.0])

        def area_properties(self):
            """Calculates the area and centroid of the slice

            Returns
            -------
            area : float
                The internal area
            centroid : np.array
                The centroid of the area
            """
            if len(self.edges) == 0:
                return 0, np.array([0.0, 0.0, 0.0])
            x0 = (self.vertices[self.edges[:, 0]] - self.centre) @ self.x
            x0 = (self.vertices[self.edges[:, 0]] - self.centre) @ self.x
            x0 = (self.vertices[self.edges[:, 0]] - self.centre) @ self.x
            y0 = (self.vertices[self.edges[:, 0]] - self.centre) @ self.y
            x1 = (self.vertices[self.edges[:, 1]] - self.centre) @ self.x
            y1 = (self.vertices[self.edges[:, 1]] - self.centre) @ self.y

            t = np.zeros((self.edges.shape[0], 3, 3))
            t[:, 0, 2] = t[:, 1, 2] = t[:, 2, 2] = 1.0
            t[:, 1, 0] = x0
            t[:, 1, 1] = y0
            t[:, 2, 0] = x1
            t[:, 2, 1] = y1

            areas = (
                np.linalg.det(t) / 2.0
            )  # np.array([0.5 * np.linalg.det(np.array([[0.,0.,1.],[x0[i], y0[i], 1.], [x1[i], y1[i], 1.]])) for i in range(len(self.edges))])
            centroids = (
                self.vertices[self.edges].sum(axis=1) + self.centre
            ) / 3.0  # np.array([(self.vertices[edge[0]] + self.vertices[edge[1]] + self.centre) / 3.0 for edge in self.edges])
            area = sum(areas)
            if area != 0:
                centroid = areas @ centroids / area
            else:
                centroid = np.array([0.0, 0.0, 0.0])
            return area, centroid

        def moment_area(self):
            """Calculates the first moments of area

            Returns
            -------
            Ax : float
                The moment of area in x
            Ay : float
                The moment of area in y
            """
            if len(self.edges) == 0:
                return 0, 0
            x0 = (self.vertices[self.edges[:, 0]] - self.centre) @ self.x
            x0 = (self.vertices[self.edges[:, 0]] - self.centre) @ self.x
            x0 = (self.vertices[self.edges[:, 0]] - self.centre) @ self.x
            y0 = (self.vertices[self.edges[:, 0]] - self.centre) @ self.y
            x1 = (self.vertices[self.edges[:, 1]] - self.centre) @ self.x
            y1 = (self.vertices[self.edges[:, 1]] - self.centre) @ self.y

            XD = x1 - x0
            YD = y1 - y0

            Ax = -XD / 2 * (y0 ** 2 + YD * (y0 + YD / 3))
            Ay = YD / 2 * (x0 ** 2 + XD * (x0 + XD / 3))

            return sum(Ax), sum(Ay)

        def moment_inertia(self):
            """Calculates the moments of inertia

            Also referred to as the second moment of area, as we have constant density

            Returns
            -------
            Ixx : float
                The moment of inertia in x
            Iyy : float
                The moment of inertia in y
            Ixy : float
                The product of inertia
            """
            if len(self.edges) == 0:
                return 0, 0, 0
            x0 = (self.vertices[self.edges[:, 0]] - self.centre) @ self.x
            x0 = (self.vertices[self.edges[:, 0]] - self.centre) @ self.x
            x0 = (self.vertices[self.edges[:, 0]] - self.centre) @ self.x
            y0 = (self.vertices[self.edges[:, 0]] - self.centre) @ self.y
            x1 = (self.vertices[self.edges[:, 1]] - self.centre) @ self.x
            y1 = (self.vertices[self.edges[:, 1]] - self.centre) @ self.y

            Ixx = (
                x0 * y0 ** 2 * y1 / 12
                + x0 * y0 * y1 ** 2 / 12
                + x0 * y1 ** 3 / 12
                - x1 * y0 ** 3 / 12
                - x1 * y0 ** 2 * y1 / 12
                - x1 * y0 * y1 ** 2 / 12
            )
            Iyy = (
                x0 ** 3 * y1 / 12
                - x0 ** 2 * x1 * y0 / 12
                + x0 ** 2 * x1 * y1 / 12
                - x0 * x1 ** 2 * y0 / 12
                + x0 * x1 ** 2 * y1 / 12
                - x1 ** 3 * y0 / 12
            )
            Ixy = (
                x0 ** 2 * y0 * y1 / 12
                + x0 ** 2 * y1 ** 2 / 24
                - x0 * x1 * y0 ** 2 / 12
                + x0 * x1 * y1 ** 2 / 12
                - x1 ** 2 * y0 ** 2 / 24
                - x1 ** 2 * y0 * y1 / 12
            )

            return sum(Ixx), sum(Iyy), sum(Ixy)


if __name__ == "__main__":
    # t = trimesh.load('C:/Users/R_D_Student/Documents/Part1.stl')
    t = list(
        trimesh.load(
            "C:\\Users\\R_D_Student\\Downloads\\boat_v1_L3.123c4b06c27e-11c8-4fd1-9350-17fb64b49970\\boat_v1_L3.123c4b06c27e-11c8-4fd1-9350-17fb64b49970\\11806_boat_v1_L3.obj"
        ).geometry.values()
    )[1]
    vertices = np.array(t.vertices)
    faces = np.array(t.faces)
    """
    vertices = np.array([
        [0.,0.,0.],
        [0.,0.,1.],
        [0.,1.,1.],
        [0.,1.,0.],
        [1.,0.,0.],
        [1.,0.,1.],
        [1.,1.,1.],
        [1.,1.,0.]
    ])
    faces = np.array([
        [0,1,2],
        [0,2,3],
        [4,6,5],
        [4,7,6],
        [1,5,6],
        [1,6,2],
        [2,6,3],
        [3,6,7],
        [0,3,4],
        [3,7,4],
        [0,4,5],
        [0,5,1]
    ])"""
    v, e, f = clip_points(
        vertices,
        faces,
        position=np.array([0.0, 25, 30]),
        normal=np.array([0.0, 0.0, 1.0]),
    )
    v, index = np.unique(v.round(decimals=4), return_inverse=True, axis=0)
    f = index[f]
    e = index[e]
    a, b = cull_interior_edges(e, v)
    f = cull_interior_faces(f, b)

    p = pv.Plotter()
    p.add_mesh(Mesh(v, f).visualise())
    print(len(a))
    # for i,edge in enumerate(a):
    #    p.add_mesh(pv.Line(pointa=v[edge[0]], pointb=v[edge[1]]))
    # p.add_mesh(pv.Arrow(start=v[edge[0]], direction=v[edge[1]]-v[edge[0]]))
    p.show()
    """
    m = Mesh(vertices, faces)
    v,c = m.volume_properties()
    a = m.area()
    print(v,c,a)
    m2, s = m.cut(np.array([0.5,10,0.5]), np.array([0.,0.,1.]), include_interior=False)
    v,c = m2.volume_properties()
    a = m.area()
    print(v,c,a)
    a,c = s.area_properties()
    A = s.moment_area()
    I = s.moment_inertia()
    print(a,c,A,I)
    p = pv.Plotter()
    p.add_mesh(m2.visualise())
    p.show()
    """
