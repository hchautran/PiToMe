# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from .sam import apply_patch as sam
from .sam2 import apply_patch as sam2
from .sam3 import apply_patch as sam3

__all__ = ['sam', 'sam2', 'sam3']
