import os
from setuptools import setup, find_packages, Command, Extension


__version__ = None
exec(open('anndataview/version.py').read())


class CleanCommand(Command):
    """
    Custom clean command to tidy up the project root.
    """
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info ./htmlcov')


setup(
    name='anndataview',
    version=__version__,
    description='Extension of the AnnData object for annotated views.',
    python_requires='>3.7.0',
    setup_requires=[
        'setuptools>=18.0',
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy>=1.16.0,<2.0.0',
        'pandas>=0.15.0',
        'anndata>=0.8.0',
        'future',
        'deepdiff',
        'h5py',
    ],
    cmdclass={
        'clean': CleanCommand
    },
)
