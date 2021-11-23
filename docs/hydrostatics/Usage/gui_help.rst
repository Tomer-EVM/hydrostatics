GUI Help
========

Menus
-----

File
^^^^
.. warning::
    These options are untested

The File menu includes options to save and load projects.

Help
^^^^
The Help memu includes one option, to view documentation.
This opens this document in the results viewer.

Meshes
------
This box provides interaction with mesh objects
Meshes can be added, renamed, repositioned and removed.

Repositioning works independently of any other calculations.

When loaded, the new mesh may not be selected. It should be available for selection in the drop-down box.

Weights
-------
This box provides interaction with weights.

Again, repositioning is independent. The magnitude should be given in Newtons.

Weights are added with their name as the current number of weights. This means that if a weight is named to a number
greater than the number of weights, and weights are added, the weight will eventually be overwritten.
Good practice would be to give any important weights proper names.

Solver
------
This box provides functions for computing hydrostatic equilibrium.

Output will be given as the solver runs, in the box below.

Unfortunately, true multithreading is not possible with python's GIL, and so the 3D view will steal
work from the solver. To speed up calculations, open the results tab and avoid computations while the solver is running.

See Equilibrium Help for more info on how to use these methods.

Other
-----
The other controls are for setting the position of the waterplane
relative to the boat, and some global constants.

Viewer
------
The Viewer tab provides a 3D view of the objects currently loaded.

The display of some objects can be toggled using the checkboxes at the bottom,
as well as in there own section for meshes and weights.

.. warning::
    Refresh View may sometimes provide an out-of-date view. It may also not display some meshes. These problems appear to be an
    internal pyvista issue. Recalculating results and refreshing the view may help.

Results
-------
The Results tab is an html viewer. A formatted html string is produced and sent to the window directly whenever
the results are recalculated. 

Clicking Full Report augments this with an image of the current view, a title, and
the values of global variables. This can be saved to a pdf file.