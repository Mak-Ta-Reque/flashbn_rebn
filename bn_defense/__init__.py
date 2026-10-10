"""BN-only backdoor defenses: FlashBN (training-free BN re-initialization) and ReBN
(FlashBN + retraining of the BN scale/shift only)."""
import os

# TensorBoard (older versions) cannot use the C++ protobuf 4+/5+ backend; the pure-Python one works.
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

from .data import AttackedModel  # noqa: E402
from .flashbn import FlashBN  # noqa: E402
from .rebn import ReBN  # noqa: E402
