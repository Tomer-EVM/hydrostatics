import os

try:
    from Cython.Build import cythonize
except ImportError:

    def build(setup_kwargs):
        pass


else:
    from setuptools import Extension
    from setuptools.dist import Distribution
    from distutils.command.build_ext import build_ext
    import numpy

    def build(setup_kwargs):
        cython_extensions = [
            "hydrostatics/fast_calcs/volume_calcs.pyx",
            "hydrostatics/fast_calcs/clipping_calcs.pyx",
        ]
        cpp_extensions = []

        # gcc arguments hack: enable optimizations
        os.environ["CFLAGS"] = "-O3"

        # Build
        setup_kwargs.update(
            {
                "ext_modules": cythonize(
                    cython_extensions,
                    language_level=3,
                    compiler_directives={"linetrace": True},
                )
                + cpp_extensions,
                "install_requires": [
                    "numpy",
                ],
                "include_dirs": [numpy.get_include()],
                "zip_safe": False,
                "cmdclass": {"build_ext": build_ext},
            }
        )
