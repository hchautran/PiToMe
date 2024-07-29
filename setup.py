# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from setuptools import find_packages, setup

setup(
    name="pitome",
    version="0.1",
    author="Meta",
    url="https://github.com/hchautran/PiToMe",
    description="Token Merging with Spectrum Preservation",
    install_requires=[
        "salesforce-lavis",
        "datasets",
        "accelerate",
        # "timm==0.6.13",
        "tokenizers==0.15.1",
        "transformers==4.37.0",
    ],
    packages=find_packages(exclude=("examples", "build")),
)
