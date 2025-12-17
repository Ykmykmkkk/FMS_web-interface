from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'fms_server'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'static'),
         glob('fms_server/static/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kyeongmin',
    maintainer_email='kyeongmin@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'fms_app = fms_server.fms_app:main',
        ],
    },
)
