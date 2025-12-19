from setuptools import setup, find_packages

setup(
    name="libreyolo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["Pillow>=8.0.0", "numpy>=1.19.0"],
)

