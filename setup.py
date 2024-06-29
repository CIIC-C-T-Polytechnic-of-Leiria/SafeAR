"""
File: setup.py
Author: Tiago F.R. Ribeiro
Last Modified: 2024-06-29
Description: 
    This setup.py file is used to configure the metadata and dependencies for the SafeAR
    - Obfuscation Service project. It uses setuptools to define the project's name, version, 
    packages, dependencies, and entry points.
"""
from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Setup configuration
setup(
    name="SafeAR - Obfuscation Service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "obfuscate = safeAR_obfuscation_aaS.cli:main"
        ],
    },
)

