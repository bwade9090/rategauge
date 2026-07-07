"""Configuration: model registry (configs/models.yaml) and API credentials (.env)."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

MODELS_PATH = Path("configs/models.yaml")


@dataclass(frozen=True)
class ModelConfig:
    key: str
    provider: str  # "openai" | "anthropic"
    model_id: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    batch_discount: float

    def cost_usd(self, input_tokens: int, output_tokens: int, *, batch: bool = False) -> float:
        cost = (
            input_tokens * self.input_usd_per_mtok + output_tokens * self.output_usd_per_mtok
        ) / 1_000_000
        return cost * (self.batch_discount if batch else 1.0)


def load_models(path: Path = MODELS_PATH) -> dict[str, ModelConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    discount = float(raw["batch_discount"])
    return {
        key: ModelConfig(
            key=key,
            provider=spec["provider"],
            model_id=spec["model_id"],
            input_usd_per_mtok=float(spec["input_usd_per_mtok"]),
            output_usd_per_mtok=float(spec["output_usd_per_mtok"]),
            batch_discount=discount,
        )
        for key, spec in raw["models"].items()
    }


def load_credentials() -> None:
    """Load API keys from .env into the environment (never logged, never printed)."""
    load_dotenv()
