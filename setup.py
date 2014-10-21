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
from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext
from distutils import sysconfig
from distutils.command.clean import clean
import os
from os.path import join


class Clean(clean):
    def run(self):
        clean.run(self)
        if self.all:
            for dir, dirs, files in os.walk('SimpleGP'):
                abspath = os.path.abspath(dir)
                for f in files:
                    ext = f.split('.')[-1]
                    if ext in ['c', 'so']:
                        os.unlink(os.path.join(abspath, f))


# -mno-fused-madd

lst = ['CFLAGS', 'CONFIG_ARGS', 'LIBTOOL', 'PY_CFLAGS']
for k, v in zip(lst, sysconfig.get_config_vars(*lst)):
    if v is None:
        continue
    v = v.replace('-mno-fused-madd', '')
    os.environ[k] = v
ext_modules = [Extension("SimpleGP.EA_aux_functions",
                         [join("SimpleGP", "aux_functions.pyx")],
                         # libraries=["m"],
                         include_dirs=[numpy.get_include()]),
               Extension("SimpleGP.Simplify_mod",
                         [join("SimpleGP", "simplify.pyx")],
                         # libraries=["m"],
                         include_dirs=[numpy.get_include()]),
               Extension("SimpleGP.Rprop_mod",
                         [join("SimpleGP", "rprop.pyx")],
                         # libraries=["m"],
                         include_dirs=[numpy.get_include()]),
               Extension("SimpleGP.RecursiveGP_mod",
                         [join("SimpleGP", "recursive.pyx")],
                         # libraries=["m"],
                         include_dirs=[numpy.get_include()]),
               Extension("SimpleGP.eval",
                         [join("SimpleGP", "eval.pxd"),
                          join("SimpleGP", "eval.pyx")],
                         # libraries=["m"],
                         include_dirs=[numpy.get_include()]),
               Extension("SimpleGP.tree",
                         [join("SimpleGP", "tree.pyx")],
                         # libraries=["m"],
                         include_dirs=[numpy.get_include()]),
               Extension("SimpleGP.pde",
                         [join("SimpleGP", "pde.pxd"),
                          join("SimpleGP", "pde.pyx")],
                         # libraries=["m"],
                         include_dirs=[numpy.get_include()])]

version = open("VERSION").readline().lstrip().rstrip()
lst = open(join("SimpleGP", "__init__.py")).readlines()
for k in range(len(lst)):
    v = lst[k]
    if v.count("__version__"):
        lst[k] = "__version__ = '%s'\n" % version
with open(join("SimpleGP", "__init__.py"), "w") as fpt:
    fpt.write("".join(lst))

setup(
    name="SimpleGP",
    description="""A GP systems for symbolic regression and
    auto-recursive regression""",
    version=version,
    url='http://dep.fie.umich.mx/~mgraffg',
    author="Mario Graff",
    author_email="mgraffg@dep.fie.umich.mx",
    cmdclass={"build_ext": build_ext, "clean": Clean},
    ext_modules=ext_modules,
    packages=['SimpleGP'],
    install_requires=['cython >= 0.19.2', 'numpy >= 1.6.2',
                      'pymock >= 1.0.5']
)










