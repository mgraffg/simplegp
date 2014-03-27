# Copyright 2013 Mario Graff Guerrero

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils import sysconfig
import os

# -mno-fused-madd

lst = ['CFLAGS', 'CONFIG_ARGS', 'LIBTOOL', 'PY_CFLAGS']
for k, v in zip(lst, sysconfig.get_config_vars(*lst)):
    v = v.replace('-mno-fused-madd', '')
    os.environ[k] = v
ext_modules = [Extension("SimpleGP.EA_aux_functions",
                         ["SimpleGP/aux_functions.pyx"],
                         libraries=["m"],
                         include_dirs=[numpy.get_include()]),
               Extension("SimpleGP.Simplify_mod",
                         ["SimpleGP/simplify.pyx"],
                         libraries=["m"],
                         include_dirs=[numpy.get_include()]),
               Extension("SimpleGP.Rprop_mod",
                         ["SimpleGP/rprop.pyx"],
                         libraries=["m"],
                         include_dirs=[numpy.get_include()]),
               Extension("SimpleGP.RecursiveGP_mod",
                         ["SimpleGP/recursive.pyx"],
                         libraries=["m"],
                         include_dirs=[numpy.get_include()]),
               Extension("SimpleGP.eval",
                         ["SimpleGP/eval.pxd",
                          "SimpleGP/eval.pyx"],
                         libraries=["m"],
                         include_dirs=[numpy.get_include()]),
               Extension("SimpleGP.tree",
                         ["SimpleGP/tree.pyx"],
                         libraries=["m"],
                         include_dirs=[numpy.get_include()])]

setup(
    name="SimpleGP",
    description="""A GP systems for symbolic regression and
    auto-recursive regression""",
    version="0.2.7",
    url='http://dep.fie.umich.mx/~mgraffg',
    author="Mario Graff",
    author_email="mgraffg@dep.fie.umich.mx",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    packages=['SimpleGP',
              'SimpleGP.tests']
)
