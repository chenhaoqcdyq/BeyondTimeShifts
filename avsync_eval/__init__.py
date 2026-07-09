"""AV-Sync evaluator package.

The bundled `qwen2_5_omni` and `qwen2_vl` implementations live inside this
package directory and are imported with top-level names
(e.g. `from qwen2_5_omni.modeling_qwen2_5_omni import ...`). Add this directory
to sys.path on import so those names resolve regardless of the entry point.
"""
import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
