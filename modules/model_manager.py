"""
Model lifecycle utilities for the Flask API.

This module defines :class:`ModelManager`, a thin wrapper around
:class:`modules.inference.Inference` that centralises model loading,
unloading, and access.

Key features
------------
* **Lazy loading** – a model is only loaded the first time
  :py:meth:`ModelManager.load` is called.
* **Safe unloading** – :py:meth:`ModelManager.unload` frees GPU VRAM,
  clears Python references, and resets the manager.
* **Reusable access** – :py:meth:`ModelManager.get_model` returns the
  active model instance (or ``None``) for downstream components.
"""

import gc
from typing import Any

import torch
from api import settings

from modules.inference import Inference


class ModelManager:
    """
    Manages loading, unloading, and accessing the current language model.

    This class encapsulates all model lifecycle operations to avoid the use
    of global variables and to enable easier testing and state management.
    """
    def __init__(self) -> None:
        """
        Initialize the ModelManager with no model loaded.
        """
        self.inference: Inference | None = None

    def load(self, model_name: str) -> None:
        """
        Load the specified model into memory if not already loaded.

        Args:
            model_name: The name of the model to load from the model directory.
        """
        if self.inference:
            return

        settings.model_name = model_name
        model_path = f"{settings.model_dir}/{model_name}"

        self.inference = Inference(model_path=model_path, quantize=True)
        self.inference.load_model()

    def unload(self) -> None:
        """
        Unload the currently loaded model from memory and clear GPU VRAM.
        """
        if self.inference:
            del self.inference.model
            self.inference = None
            gc.collect()
            torch.cuda.empty_cache()

    def get_model(self) -> Any | None:
        """
        Return the underlying model instance, if available.

        Returns:
            The loaded model object, or None if no model is loaded.
        """
        return self.inference.model if self.inference else None
