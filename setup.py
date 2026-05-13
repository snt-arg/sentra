import os
from glob import glob
from setuptools import setup, find_packages

pkg = "sentra_ros"


def parseRequirements(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    reqs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


setup(
    name=pkg,
    zip_safe=True,
    version="1.0.0",
    license="GPLv3",
    author="Ali Tourani",
    maintainer="Ali Tourani",
    python_requires=">=3.12",
    include_package_data=True,
    url="https://github.com/snt-arg/sentra",
    packages=find_packages(exclude=["docs"]),
    maintainer_email="a.tourani1991@gmail.com",
    long_description_content_type="text/markdown",
    long_description=open("README.md", encoding="utf-8").read(),
    description="⚜️ GenAI-powered vision-language grounding for vS-Graphs ⚜️",
    entry_points={
        "console_scripts": ["sentra_node=sentra_ros.sentra:main"],
    },
    install_requires=parseRequirements("src/requirements.txt"),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + pkg]),
        ("share/" + pkg, ["package.xml"]),
        (os.path.join("share", pkg, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", pkg, "config"), glob("config/*.yml")),
    ],
)
