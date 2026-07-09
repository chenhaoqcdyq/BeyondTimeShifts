# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

# Simplified __init__ for standalone usage outside the Hugging Face transformers
# package layout. Import modules directly without relying on transformers.utils.

if TYPE_CHECKING:
    from .configuration_qwen2_5_omni import *
    from .modeling_qwen2_5_omni import *
    from .processing_qwen2_5_omni import *
else:
    # Eagerly import symbols to make the package usable without transformers' lazy loader
    from .configuration_qwen2_5_omni import *  # noqa: F401,F403
    from .modeling_qwen2_5_omni import *  # noqa: F401,F403
    from .processing_qwen2_5_omni import *  # noqa: F401,F403
