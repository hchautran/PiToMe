# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from .deit import apply_patch as deit
from .mae import apply_patch as mae
from .aug import apply_patch as aug 
from .blip import apply_patch as blip 
# from .blip2 import apply_patch as blip2
# from .clip import apply_patch as clip

__all__ = [
    "deit",
    "mae",
    "aug", 
    "blip", 
    # "blip2", 
    # "clip" 
]
