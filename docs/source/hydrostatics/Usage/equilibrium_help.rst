Static Equilibrium
==================

There are currently five methods implemented for finding hydrostatic equilibrium.

Iterative
---------

This method uses an sort of semi-Newtons method for finding the equilibrium. It was implemented in the
original spreadsheet. The core logic is the same, although there are some differences in implementation.

This method works relatively well, especially when the starting point is near to the equilibrium.

Iterative Multidimensional
--------------------------

This method is an extension of the iterative method, utilising approximations for the change in force when heel and trim are changed,
and for change in moments when height is changed. It can be thought of as Newtons multidimensional root finding method, although there are
some differences.

This method tends to converge faster than the iterative method, but only slightly. Either will work well

Iterative force
---------------

This is essentially a gradient descent algorithm, where the force and moments are taken to be the gradient. It could also be thought of as a very simple 
simulator, although it isn't truely dynamic.

It is useful for verifying other parts of the software work as expected, and can also be used to find the equilibrium. However, it is very slow in comparison
to other methods.

IpOpt Methods
-------------

There are two methods using IpOpt. They are a work in progress, and not very functional. If IpOpt is not installed, they will not be loaded.
