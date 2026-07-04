from setuptools import setup, find_packages

setup(
    name="phylogriffin",
    version="0.3.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "dendropy>=4.6.0",
        "tqdm>=4.65.0",
    ],
)
