from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['lane_detection'],
    package_dir={'': 'src'}
)
setup(**d)
