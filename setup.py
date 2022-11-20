#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

extras = {
    "dev": [
        "pre-commit==2.20.0",
        "pre-commit-hooks==4.3.0",
        "pyupgrade==3.2.2",
        "black==22.10.0",
        "isort==5.10.1",
        "mypy==0.991",
        "mypy-extensions==0.4.3",
        "types-PyYAML==6.0.12.2",
        "types-python-dateutil==2.8.19.4",
        "pytest==6.2.5",
    ],
}

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
    python_requires=">=3.8",
    install_requires=[
        "torch==1.13.0",
        "torchvision==0.14.0",
        "pytorch-lightning==1.7.7",
        "torchmetrics==0.10.3",
        "lightning-flash==0.8.1",
        "timm==0.6.11",
        "wandb==0.13.5",
        "torch_optimizer==0.3.0",
        "pytorch-metric-learning==1.6.3",
        "rich",
        "datasets[vision]==2.2.1",
    ],
    extras_require=extras,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
)
