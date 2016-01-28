try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description'      : 'Fourier optics-based simulations for microlens arrays (MLAs),
    'long_description' : 'SimMLA provides routines for simulating microlens arrays and flys eye condensers.',
    'author'           : 'Kyle M. Douglass',
    'url'              : 'https://github.com/kmdouglass/STORMlikeChains',
    'download_url'     : 'https://github.com/kmdouglass/STORMlikeChains',
    'author_email'     : 'kyle.douglass@epfl.ch',
    'version'          : '0.1.0a1',
    'install_requires' : ['numpy',
                          'scipy'],
    'packages'         : ['SimMLA'],
    'scripts'          : [],
    'name'             : 'SimMLA',
    'classifiers'      : ['Development Status :: 3 - Alpha',
                          'Programming Language :: Python :: 3',
                          'Intended Audience :: Science/Research',
                          'Topic :: Scientific/Engineering'],
    'keywords'         : 'fourier optics microlens lenslet array microscopy'
    
}

setup(**config)
