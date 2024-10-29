# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from setuptools import find_packages, setup

setup(
    name="pitome",
    version="0.1",
    author="Hoai-Chau Tran",
    url="https://github.com/hchautran/PiToMe",
    description="Token Merging with Spectrum Preservation",
    install_requires=[
        "salesforce-lavis",
        "datasets",
        "accelerate",
        "tokenizers==0.15.1",
        "transformers==4.37.0",
        "timm==0.4.12",
        "ml-collections",
        "numpy==1.26.4",
        "pandas==2.2.1",
        "wandb"
    ],
    packages=find_packages(exclude=("data", "log", "notebooks", "tasks", "scripts", "build")),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license = 'MIT',
)
