from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("contact_map_cython.pyx",
                          language_level="3",
                          compiler_directives={
                              "boundscheck": False,
                              "wraparound": False,
                              "cdivision": True
                          }),
    include_dirs=[numpy.get_include()]
)
