Buoyancy Model
==============


This module contains the main api for creating buoancy models.

Initialising and Loading Data
-----------------------------

The buoyancy model can be initialised by:

>>> from hydrostatics.models import BuoyancyModel
>>> m = BuoyancyModel()

To load meshes into the model, we use:

>>> m.load_mesh('path/to/file')

For dxf files, we may wish to specify the shape of the uv mesh. If we do not, the shape dimensions will be set as square as possible.

>>> m.load_mesh('path/to/file.dxf', shape=(33,33))

The resulting mesh objects are stored in a dict under `m.meshes`. The name to index a mesh will be the filename, without the full path. dxf files have numbers appended, as it is possible for multiple meshes to be in one file. Therefore, the mesh in the above example can be found through:

>>> m.meshes['file.dxf_0']

To add weights to the model, we use:

>>> m.set_weight_force('name of weight', position=np.array([x,y,z]), value=weight_force)

Note that the weight force should be given in Newtons.

This is a thin layer over a dictionary assignment. If we need to change the position or value, the weight data can be found in:

>>> m.weight_forces['name of weight']

Transformations
---------------

We can specify rotations and translations for individual meshes. Using the same naming scheme as before:

>>> m.local_rotation['file.dxf_0'] = np.array([x,y,z])
>>> m.local_position['file.dxf_0'] = np.array([x,y,z])

This will not immediately apply transformations. To do this, use:

>>> m.calculate_transformation()

This will apply rotation and translation, in that order.

Further manipulation
^^^^^^^^^^^^^^^^^^^^

dxf files are commonly given as half a mesh, and need mirroring. Also, they are sometimes not watertight, and need ends closed off. To apply these operations, we can do:

>>> m.mirror('file.dxf_0')
>>> m.close_ends('file.dxf_0')

These operations will not work on non-dxf meshes. Other meshes are imported using Trimesh, so see that documentation for further information.

Waterplane
----------

To fully define the problem, we also need to specify the position of the waterplane. This can be done in a few ways.

**1.**

We can set the water plane position through heel, trim and height:

>>> m.set_waterplane(heel, trim, height)

By default the rotation will be about the origin. If we wish to change this, we can set the centre of rotation to a different value:

>>> m.set_waterplane(heel, trim, height, cor=np.array([10,10,10]))

**2.**

The above method uses the rotation and height to calculate a normal and a origin for the plane. We cannot set the normal directly, but we can specify the origin:

>>> m.heel = 90
>>> m.trim = 45
>>> m.waterplane_origin = np.array([10,10,10])

Force Calculation
-----------------

Once the problem is fully specified, we can calculate forces using the run() method:

>>> forces, moments = m.run()

The forces and moments are 3d vectors in x,y,z.

The reference frame for the forces by default is the body reference frame. For some applications the earth reference frame is more useful, as the x and y forces, and the z moment, will be 0:

>>> forces, moments = m.run(reference='earth')

We can also give the reference as a rotation from the earth reference:

>>> R = np.array([
...     [0,1,0],
...     [1,0,0],
...     [0,0,1]
... ])
>>> forces, moments = m.run(reference=R)


Further results
---------------

A variety of extra information can also be calculated. After using the .run() method, we can retrieve the volume displaced, centre of buoancy, centre of gravity, weight forces and buoancy forces in both reference frames. As an example:

>>> m.run()
>>> m.results.force
>>> m.results.weight_moment
>>> m.results.volume_centroid

To produce further results, we can call .calculate_results(). All relevant results are stored under .results. This has a nice print output.

>>> m.calculate_results()
>>> print(m.results)
    Forces:
    Buoyancy:   x = 0.0000e+00, y = 0.0000e+00, z = -3.6485e+04
    Weight:     x = 0.0000e+00, y = 0.0000e+00, z = -1.0000e+01
    Total:      x = 0.0000e+00, y = 0.0000e+00, z = -3.6495e+04
    Moments:
    Buoyancy    x = 1.0869e+08, y = 1.7841e+08, z = 0.0000e+00
    Weight      x = 0.0000e+00, y = 0.0000e+00, z = 0.0000e+00
    Total:      x = 1.0869e+08, y = 1.7841e+08, z = 0.0000e+00
    Volume = -3628420610.9277
    Center of Buoyancy:
        x = 4889.8876, y = -2979.0009, z = -80.5666
    Center of Gravity:
        x = 0.0000, y = 0.0000, z = 0.0000
    Wetted Surface Area = 5068903.7727
    Waterplane Area = -29403713.5379
    Center of Flotation:
        x = 4818.5768, y = -2675.5666, z = 0.0000
    Moment of Inertia:
        Ixx = -9478798271947.4648, Iyy = -126203033160314.5938, Ixy = 17236853785882.5234
        Iu = -9478798271947.3203, Iv = -126203033160314.7500, Theta = -36.7729
    Length Water Line = 13659.6182
    Beam Water Line = 345.0432
    Waterplane Coefficient = -6.2386
    BMt = 0.0000, BMl = 0.0000
    GMt = 0.0000, GMl = 0.0000

The results include wetted surface area, moments of area and inertia of the waterplane, the length and width of the waterplane and the waterplane coefficient.

Plotting
--------

3d plots are supported through pyvista. An example:

>>> import pyvista as pv
>>> p = pv.Plotter()
>>> for mesh in m.plot_transformed(): 
...     p.add_mesh(mesh)
>>> for mesh in m.plot_below_surface():
...     p.add_mesh(mesh)
>>> for mesh in m.plot_weights(): 
...     p.add_mesh(mesh)
>>> p.add_mesh(m.plot_waterplane())
>>> p.show()


Saving
------

We can either save meshes, or the entire model.

A save method is implemented for the `Mesh` class. It converts the object to a trimesh object to save meshes, so for full details see the trimesh API.
An example of it being used:

>>> model.transformed['my.obj'].save('my_output.stl')
>>> model.faces_below_water['my.obj'].save('my_cut.stl')

It is also possible to save the entire model. This is a thin wrapper over the Pickle library, and is mostly meant for the GUI. It can still be used as follows:

>>> model.save('path/to/backup.hydro')
>>> backup = load_hydro('path/to/backup.hydro')