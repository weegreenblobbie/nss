from setuptools import setup, Extension
import pybind11
import numpy
import os
import shutil
from distutils.dir_util import remove_tree
from distutils.core import Command

class clean(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        for root, dirs, files in os.walk(os.path.abspath(os.path.dirname(__file__))):
            for f in files:
                if f.endswith(".so") or f.endswith(".pyd"):
                    os.unlink(os.path.join(root, f))
        remove_tree(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'build'), verbose=1)

setup(
    cmdclass={'clean': clean},
    ext_modules=[
        Extension(
            "utils_cpp",
            [
                "nss/sorted_container.cpp",
                "nss/squash.cpp",
            ],
            include_dirs=[
                "nss",
                pybind11.get_include(),
                numpy.get_include()
            ],
            language="c++"
        ),
    ],
    zip_safe=False,
)