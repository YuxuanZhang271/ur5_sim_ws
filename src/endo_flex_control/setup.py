from setuptools import setup
import os
from glob import glob

package_name = 'endo_flex_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'urdf'),   glob('urdf/*.urdf') + glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'),   glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Endoscope flexible section control package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'endoflex_teleop = endo_flex_control.endo_flex_control:main',
        ],
    },
)