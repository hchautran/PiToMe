# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from .swag import apply_patch as swag
from .deit import apply_patch as deit
from .aug import apply_patch as aug 
from .mae  import apply_patch as mae

__all__ = ["deit", "swag", "mae", "aug"]
