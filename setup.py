from setuptools import setup

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="movie_classifier",
    packages=["movie_classifier"],
    version="0.0.1",
    entry_points={"console_scripts": ["movie_classifier=movie_classifier.cli:main"]},
    install_requires=required_packages,
)
