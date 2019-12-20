from setuptools import setup

setup(
    name='df2markov',
    version='0.1',
    description='A simple way to create markov chains from dataframes',
    author='Susan Vermeer and Damian Trilling',
    author_email='s.a.m.vermeer@uva.nl',
    url='https://github.com/uvacw/df2markov',
    packages=['df2markov'],  
    install_requires=['pandas', 'pydot', 'numpy', 'networkx'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Sociology',
                 'Programming Language :: Python :: 3'],
    keywords = ['markov chain','timestamped data','social science'],
)
