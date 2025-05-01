from setuptools import find_packages, setup

import os


from Cython.Build import cythonize

from distutils.command.build_ext import build_ext
import numpy


cython_extensions = [
    "hydrostatics/fast_calcs/volume_calcs.pyx",
    "hydrostatics/fast_calcs/clipping_calcs.pyx",
]
cpp_extensions = []

# gcc arguments hack: enable optimizations
os.environ["CFLAGS"] = "-O3"


setup(
    packages=find_packages(),
    ext_modules=cythonize(
        cython_extensions,
        language_level=3,
        compiler_directives={"linetrace": True},
    )
    + cpp_extensions,
    install_requires=[
        "numpy",
    ],
    include_dirs=[numpy.get_include()],
    zip_safe=False,
    cmdclass={"build_ext": build_ext},
)
print("b")
