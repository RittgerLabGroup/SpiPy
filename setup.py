#!/usr/bin/env/python

import os

import numpy
import setuptools
from setuptools.command.build_py import build_py as _build_py


conda_prefix = os.environ.get("CONDA_PREFIX", "/usr")  # fallback if not in conda

def _existing_dirs(paths):
    return [path for path in paths if os.path.isdir(path)]


if os.environ.get("CONDA_PREFIX"):
    # In Conda builds, prefer the active environment and avoid mixing in
    # system headers that can conflict with the env toolchain.
    NLOP_LIB_DIRS = _existing_dirs([
        os.path.join(conda_prefix, 'lib'),
    ])
    NLOP_INCLUDE_DIRS = _existing_dirs([
        'include',
        os.path.join(conda_prefix, 'include'),
    ])
else:
    NLOP_LIB_DIRS = _existing_dirs([
        '/opt/homebrew/lib',
        '/usr/lib',
        '/usr/local/lib',
        os.path.join(conda_prefix, 'lib'),
    ])
    NLOP_INCLUDE_DIRS = _existing_dirs([
        '/opt/homebrew/include',
        '/usr/include',
        '/usr/local/include',
        'include',
        os.path.join(conda_prefix, 'include'),
    ])

INCLUDE_DIRS = NLOP_INCLUDE_DIRS + [numpy.get_include()]

spires = setuptools.Extension(  
    name='spires._core',
    sources=['spires/spires.i', 'spires/spires.cpp'],
    swig_opts=['-c++'],
    extra_compile_args=['-std=c++11'],
    library_dirs=NLOP_LIB_DIRS,    
    include_dirs=INCLUDE_DIRS,     
    libraries=['nlopt'],
    language='c++'
)


class build_py(_build_py):

    def run(self):
        """ 
        We need to overwrite run to make sure extension is built before getting copied
        """
        self.run_command("build_ext")
        return super().run()


setuptools.setup(        
    packages=setuptools.find_packages(),
    ext_modules=[spires],
)
