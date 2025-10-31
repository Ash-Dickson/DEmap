from setuptools import setup
setup(name='DEmap',
    version='0.11.0',
    url = 'https://github.com/Ash-Dickson/DEmap',
    description='Machine-learn the threshold displacement energy surface of a material with Gaussian Process Regression. ',
    author='Ashley Dickson',
    author_email='a.dickson2@lancaster.ac.uk',
    license='MIT license',
    packages=['DEmap'],
    install_requires=['gpytorch',
                      'numpy', 'torch', 'ovito', 'dataclasses', 'matplotlib', 'scipy']
)
