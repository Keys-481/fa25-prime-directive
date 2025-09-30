from setuptools import setup

package_name = "potato_mission"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["potato_mission/launch/mission.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="Rover mission state machine",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mission_node = potato_mission.mission_node:main",
        ],
    },
)
