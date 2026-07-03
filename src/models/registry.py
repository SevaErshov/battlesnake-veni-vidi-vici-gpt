from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.models.policy import PolicyModel
from src.models.value import ValueModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, artifacts_dir: Path) -> None:
        self.artifacts_dir = artifacts_dir
        self.policy: PolicyModel | None = None
        self.value: ValueModel | None = None
        self.metadata: dict[str, Any] = {}
        self.load_models()

    def load_models(self) -> None:
        policy_path = self.artifacts_dir / "policy_model.cbm"
        value_path = self.artifacts_dir / "value_model.cbm"
        metadata_path = self.artifacts_dir / "model_metadata.json"
        try:
            if metadata_path.exists():
                self.metadata = json.loads(metadata_path.read_text())
            if policy_path.exists():
                self.policy = PolicyModel.load(policy_path)
            if value_path.exists():
                self.value = ValueModel.load(value_path)
        except Exception as exc:
            logger.exception("Failed to load models: %s", exc)
            self.policy = None
            self.value = None

    def has_models(self) -> bool:
        return self.policy is not None and self.value is not None

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        metadata_path = self.artifacts_dir / "model_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
