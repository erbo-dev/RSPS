from setuptools import setup, find_packages

setup(
    name="harmkit",
    version="0.1.0",
    description="Decision-level measurement of artifact-induced BCI failures",
    packages=find_packages(include=["harmkit", "harmkit.*"]),
    python_requires=">=3.9",
    install_requires=["numpy>=1.22", "scipy>=1.8"],
    license="MIT",
)
