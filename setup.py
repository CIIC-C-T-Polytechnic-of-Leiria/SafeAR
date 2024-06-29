from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="SafeAR - Obfuscation Service",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "obfuscate = safeAR_obfuscation_aaS.cli:main",
        ],
    },
)



