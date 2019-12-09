from setuptools import setup

setup(
       name='df2markov',
       version='0.1',
       description='A simple way to create markov chains from dataframes',
       author='Susan Vermeer and Damian Trilling',
       author_email='s.a.m.vermeer@uva.nl',
       packages=['df2markov'],  
       install_requires=['pandas'] # ,  TODO: check
       # scripts=['bin/...']  # add in case we want to have a command line tool
    )
