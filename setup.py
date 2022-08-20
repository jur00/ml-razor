from setuptools import setup, find_packages

from codecs import open
from os import path

HERE = path.abspath(path.dirname(__file__))
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ml-razor",
    version="0.1.8",
    description="Tool for minimizing amount of features, while keeping model performance high.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://ml-razor.readthedocs.io/",
    author="Jurriaan Oudenaarden",
    author_email="jurriaanoudenaarden@gmail.com",
    license="MIT",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    packages=["ml_razor"],
    include_package_data=True,
    install_requires=["numpy", "pandas", "scipy", "matplotlib", "scikit-learn"]
)
