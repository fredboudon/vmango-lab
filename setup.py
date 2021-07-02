from setuptools import setup, find_packages
from glob import glob
import os

setup(
    name='vmlab',
    version='1.0.0',
    url='https://github.com/jvail/vmango-lab',
    author='Frederic Boudon, Jan Vaillant, Isabelle Grechi, Frederic Normand',
    author_email='frederic.boudon@cirad.fr',
    description='V-Mango model',
    long_description='vmlab is a simulation and analysis tool for mango tree architecture.',
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
