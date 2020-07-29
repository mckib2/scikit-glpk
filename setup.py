'''Install scikit-glpk bindings.'''

from setuptools import find_packages, setup
from distutils.core import Extension
import pathlib

GLPK_SRC_DIR = pathlib.Path('glpk-4.65/src')

def scrape_makefile_list(filename, START_TOKEN, END_TOKEN):
    with open(filename, 'r', encoding='utf-8') as f:
        _contents = f.read()
        sidx = _contents.find(START_TOKEN)
        eidx = _contents.find(END_TOKEN)
        lines = _contents[sidx+len(START_TOKEN):eidx].splitlines()
        return [str(_l.replace('\\', '').strip()) for _l in lines]

# Get sources for GLPK
makefile = GLPK_SRC_DIR / 'Makefile.am'
sources = scrape_makefile_list(makefile, 'libglpk_la_SOURCES = \\\n', '\n## eof ##')
sources = [str(GLPK_SRC_DIR / _s) for _s in sources]

# Get include dirs for GLPK
include_dirs = scrape_makefile_list(makefile, 'libglpk_la_CPPFLAGS = \\\n', '\nlibglpk_la_LDFLAGS')
include_dirs = [str(GLPK_SRC_DIR / _d[len('-I($srcdir)/'):]) for _d in include_dirs]


setup(
    name='scikit-glpk',
    version='0.1.0',
    author='Nicholas McKibben',
    author_email='nicholas.bgp@gmail.com',
    url='https://github.com/mckib2/scikit-glpk',
    license='MIT',
    description='Python linprog interface for GLPK',
    long_description=open('README.rst', encoding='utf-8').read(),
    packages=find_packages(),
    keywords='glpk linprog scikit',
    install_requires=open('requirements.txt').read().split(),
    python_requires='>=3.5',

    ext_modules=[
        Extension(
            'glpk4_65',
            sources=sources,
            include_dirs=include_dirs,
            language='c',
        )
    ],
)
