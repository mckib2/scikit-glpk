'''Install scikit-glpk bindings as well as GLPK library.'''

from setuptools import find_packages
import pathlib
import platform
from shutil import copy2
import subprocess

VERSION = '0.0.2'
GLPK_VERSION = '4.65'
GLPK_LIB_NAME = 'libglpk'

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('scikit-glpk', parent_package, top_path)
    config.version = VERSION

    # Either run install script for visual studio
    # or run ./configure script for Mac/Linux
    glpk_folder = pathlib.Path('src/glpk-%s' % GLPK_VERSION)
    if platform.system() == 'Windows':
        if platform.architecture()[0] == '64bit':
            subdir = 'w32'
        else:
            subdir = 'w64'
        cmd = glpk_folder / subdir / 'Build_GLPK_with_VC14_DLL.bat'
        subprocess.run([cmd], cwd=glpk_folder)
        src = str((glpk_folder / 'src' / '.libs' / GLPK_LIB_NAME).with_suffix('.dll'))
        copy2(src, 'glpk/%s.dll' % GLPK_LIB_NAME)
    elif platform.system() in ['Linux', 'Darwin']:
        subprocess.run(['./configure'], cwd=glpk_folder)
        subprocess.run(['make'], cwd=glpk_folder)
        src = str((glpk_folder / 'src' / '.libs' / GLPK_LIB_NAME).with_suffix('.so'))
        copy2(src, 'glpk/%s.so' % GLPK_LIB_NAME)
    else:
        raise OSError('Platform "%s" not supported' % platform.system())

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(
        author='Nicholas McKibben',
        author_email='nicholas.bgp@gmail.com',
        url='https://github.com/mckib2/scikit-glpk',
        license='MIT',
        description='Python linprog interface for GLPK',
        long_description=open('README.rst', encoding='utf-8').read(),
        packages=find_packages(),
        keywords='glpk linprog scikit',
        install_requires=open('requirements.txt').read().split(),
        setup_requires=['numpy'],
        python_requires='>=3.5',
        **configuration(top_path='').todict()
    )
