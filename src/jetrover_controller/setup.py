from setuptools import find_packages, setup

package_name = 'jetrover_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'librosa'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'move_robot = jetrover_controller.move_robot:main', 
        	'move_arm = jetrover_controller.move_arm:main',
        	'robo_manager = jetrover_controller.robot_manager:main',
            	'lidar_safety = jetrover_controller.lidar_safety:main',
            	'dance_attempt = jetrover_controller.dance_attempt:main',
                'buzzer_stop = jetrover_controller.buzzer_stop:main',
                'enhanced_dance = jetrover_controller.enhanced_dance:main',
                'gui = jetrover_controller.control_gui:main',
                'keyboard = jetrover_controller.keyboard:main',
                'robot_sync = jetrover_controller.robot_synchronizer:main',
                'multi_arm = jetrover_controller.multi_robot_arm_controller:main',
                'arm_sync = jetrover_controller.arm_synchronizer:main',
                'arm_keyboard = jetrover_controller.arm_keyboard:main',
                'mecanum_dance = jetrover_controller.mecanum_dance:main',
                'master_keyboard = jetrover_controller.master_keyboard:main',
                'simple_choreography = jetrover_controller.simple_choreography:main'
        ],
    },
)
