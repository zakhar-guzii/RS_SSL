import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and merge YAML configs: shared -> dataset -> model."""

    def __init__(self, config_dir: Path = Path("configs")):
        self.config_dir = config_dir
        self.shared = self._load_yaml(config_dir / "shared.yaml")

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file safely."""
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge override into base (override wins)."""
        result = base.copy()

        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def load_experiment(self, dataset_name: str, model_name: str) -> Dict[str, Any]:
        """
        Load full config: shared + dataset + model.

        Args:
            dataset_name: "uci_har", "cifar10", etc.
            model_name: "cnn_lstm", "transformer", etc.

        Returns:
            Merged config dictionary
        """
        # Load in order: shared -> dataset -> model
        config = self.shared.copy()

        # Merge dataset config
        dataset_path = self.config_dir / "datasets" / f"{dataset_name}.yaml"
        dataset_cfg = self._load_yaml(dataset_path)
        config = self._deep_merge(config, dataset_cfg)

        # Merge model config
        model_path = self.config_dir / "models" / f"{model_name}.yaml"
        model_cfg = self._load_yaml(model_path)
        config = self._deep_merge(config, model_cfg)

        # Add metadata
        config["_metadata"] = {
            "dataset": dataset_name,
            "model": model_name,
            "version": "1.0",
        }

        return config

    def save_experiment(self, config: Dict, output_path: Path) -> None:
        """Save merged config for reproducibility."""
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    loader = ConfigLoader()

    cfg = loader.load_experiment("uci_har", "cnn_lstm")

    print("=== Full Config ===")
    print(yaml.dump(cfg, default_flow_style=False))

    loader.save_experiment(cfg, Path("outputs/config_uci_har_cnn_lstm.yaml"))
