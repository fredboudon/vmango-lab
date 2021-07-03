import io
import os
from setuptools import setup, find_packages
from glob import glob
from os.path import join


def get_version(file, name='__version__'):
    """Get the version of the package from the given file by
    executing it and extracting the given `name`.
    """
    path = os.path.realpath(file)
    version_ns = {}
    with io.open(path, encoding="utf8") as f:
        exec(f.read(), {}, version_ns)
    return version_ns[name]


name = 'vmlab'
version = get_version(join(name, '_version.py'))

setup(
    name=name,
    version=version,
    url='https://github.com/jvail/vmango-lab',
    author='Frederic Boudon, Jan Vaillant, Isabelle Grechi, Frederic Normand',
    author_email='frederic.boudon@cirad.fr',
    description='A library and en environment for mango tree modeling',
    long_description='A library and an environment for the simulation and analysis of mango tree growth, development, fruit production and architecture.',
    license='Cecill-C',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'vmlab': [
            os.path.join(*path.split('/')[1:]) for path in glob('vmlab/data/**/*.*', recursive=True)
        ] + [
            os.path.join(*path.split('/')[1:]) for path in glob('vmlab/processes/*.lpy')
        ],
    }
)
