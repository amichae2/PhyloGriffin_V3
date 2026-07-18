from setuptools import find_packages, setup

setup(
    name="phylogriffin",
    version="0.3.0",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "dendropy>=4.6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "pyvolve>=1.1.0",
        "requests>=2.28.0",
    ],
)
