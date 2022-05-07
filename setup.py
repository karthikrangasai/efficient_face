#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="efficient_face",
    version="0.1.0",
    description="Describe Your Cool Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Karthik Rangasai Sivaraman, Varun Parthasarathy",
    author_email="karthikrangasai@gmail.com, varunparthasarathy7@gmail.com",
    url="https://github.com/karthikrangasai/efficient_face",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "torch==1.11.0",
        "torchvision==0.12.0",
        "pytorch-lightning==1.6.2",
        "torchmetrics==0.8.1",
        "lightning-flash==0.5.4",
        "timm==0.5.4",
        "wandb==0.12.0",
        "torch_optimizer==0.3.0",
        "pytorch-metric-learning==1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
)
