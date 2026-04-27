import os
from glob import glob
from setuptools import setup

package_name = 'lost_book_bridge'

setup(
    name=package_name,
    version='0.0.1',
    py_modules=['bridge', 'photo_camera_node', 'nav_mock'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/lost_book_bridge']),
        ('share/lost_book_bridge', ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'bridge = bridge:main',
            'photo_camera_node = photo_camera_node:main',
            'nav_mock = nav_mock:main',
        ],
    },
)
