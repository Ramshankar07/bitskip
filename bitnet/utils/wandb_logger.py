"""Optional wandb logging wrapper for BitSkip experiments."""

from typing import Dict, Optional, Any


class WandbLogger:
    """Lightweight wandb wrapper that gracefully degrades if wandb is unavailable."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: dict,
        enabled: bool = True,
        tags: Optional[list] = None,
    ):
        self.enabled = enabled
        self._run = None
        if enabled:
            try:
                import wandb

                self._run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config,
                    tags=tags,
                    reinit=True,
                )
            except ImportError:
                print("wandb not installed, logging disabled")
                self.enabled = False
            except Exception as e:
                print(f"wandb init failed: {e}, logging disabled")
                self.enabled = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if self.enabled and self._run:
            import wandb

            wandb.log(metrics, step=step)

    def summary(self, key: str, value: Any):
        if self.enabled and self._run:
            import wandb

            wandb.run.summary[key] = value

    def finish(self):
        if self.enabled and self._run:
            import wandb

            wandb.finish()
            self._run = None
