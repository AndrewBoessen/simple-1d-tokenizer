"""
Package Setup
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="simple-1d-tokenizer",
    version="0.1.0",
    author="Andrew Boessen",
    author_email="boessena@bc.edu",
    description="A image tokenizer package using vector quantization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AndrewBoessen/simple-1d-tokenizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "einops>=0.3.0",
    ],
)
