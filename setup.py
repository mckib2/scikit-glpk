'''Install scikit-glpk bindings.'''

import pathlib
from setuptools import find_packages
from distutils.core import Extension, setup
from distutils.command.build_ext import build_ext as _build_ext

GLPK_SRC_DIR = pathlib.Path('glpk-4.65/src')


def scrape_makefile_list(filename, START_TOKEN, END_TOKEN):
    '''Grab tags from GLPK makefile.'''
    with open(str(filename), 'r', encoding='utf-8') as f:
        _contents = f.read()
        sidx = _contents.find(START_TOKEN)
        eidx = _contents.find(END_TOKEN)
        lines = _contents[sidx+len(START_TOKEN):eidx].splitlines()
        return [str(_l.replace('\\', '').strip()) for _l in lines]


class build_ext(_build_ext):
    '''Overide get_export_symbols to provide them for Windows DLL.'''
    def get_export_symbols(self, ext):
        '''Only for generating Windows DLL.'''
        def_file = GLPK_SRC_DIR / '../w64/glpk_4_65.def'
        return scrape_makefile_list(def_file, 'EXPORTS\n', ';; end of file ;;')


# Get sources for GLPK
makefile = GLPK_SRC_DIR / 'Makefile.am'
sources = scrape_makefile_list(
    makefile, 'libglpk_la_SOURCES = \\\n', '\n## eof ##')
sources = [str(GLPK_SRC_DIR / _s) for _s in sources]

# Get include dirs for GLPK
include_dirs = scrape_makefile_list(
    makefile, 'libglpk_la_CPPFLAGS = \\\n', '\nlibglpk_la_LDFLAGS')
include_dirs = [
    str(GLPK_SRC_DIR / _d[len('-I($srcdir)/'):]) for _d in include_dirs]


setup(
    name='scikit-glpk',
    version='0.4.2',
    author='Nicholas McKibben',
    author_email='nicholas.bgp@gmail.com',
    url='https://github.com/mckib2/scikit-glpk',
    license='MIT',
    description='Python linprog interface for GLPK',
    long_description=open('README.rst', encoding='utf-8').read(),
    packages=find_packages(),
    keywords='glpk linprog scikit',
    install_requires=open('requirements.txt', encoding='utf-8').read().split(),
    python_requires='>=3.5',

    ext_modules=[
        Extension(
            'glpk4_65',
            sources=sources,
            include_dirs=include_dirs,
            language='c',
        )
    ],
    cmdclass={'build_ext': build_ext},
)
