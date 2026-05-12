"""Language substrate package.

Holds the Phase E kernel artifact identity layer (see ``kernel.py``).
The actual training harness and promotion governor live in
``reasoning.language_phasec`` and ``jarvis_eval.language_promotion``
respectively; this package only adds the addressable artifact wrapper
over the existing checkpoint without bypassing any existing guard.
"""

from __future__ import annotations

from .kernel import (  # noqa: F401
    LanguageKernelArtifact,
    LanguageKernelRegistry,
    get_language_kernel_registry,
)
