import codecs
import os
import re

from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="kaggle-rsna-ihd",
    version=find_version("kaggle_rsna_ihd", "__init__.py"),
    author="Araik Tamazian",
    description="A package to aid in building models for RSNA Intracranial Hemorrhage Detection task.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/atamazian/kaggle_rsna_ihd",
    packages=find_packages(exclude=["demo", "tests"]),
    install_requires=[
        "torch>=1.8",
        "torchvision",
        "pytorch-lightning>=1.5.0",
        "monai",
        "timm",
        "pandas",
        "scikit-learn>=1.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
