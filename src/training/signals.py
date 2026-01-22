"""Signal handling for graceful shutdown on RunPod Spot instances.

RunPod sends SIGTERM when a spot instance is being preempted. This module
provides a GracefulKiller class that intercepts termination signals and
allows the training loop to save state before exiting.
"""
import signal
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class GracefulKiller:
    """Signal handler for graceful shutdown on RunPod spot instances.

    RunPod sends SIGTERM when a spot instance is being preempted.
    This class catches the signal and sets a flag that the training
    loop can check to save state and exit cleanly.

    Usage:
        killer = GracefulKiller.get_instance()

        for epoch in range(num_epochs):
            if killer.should_stop():
                save_checkpoint()
                break
    """

    _instance: Optional["GracefulKiller"] = None

    def __init__(self):
        self.kill_now = False
        self.signal_received: Optional[int] = None

        # Register handlers for both SIGTERM (RunPod) and SIGINT (Ctrl+C)
        signal.signal(signal.SIGTERM, self._exit_gracefully)
        signal.signal(signal.SIGINT, self._exit_gracefully)
        logger.info("GracefulKiller initialized - listening for SIGTERM/SIGINT")

    def _exit_gracefully(self, signum: int, frame) -> None:
        """Handle termination signal by setting the kill flag."""
        signal_name = signal.Signals(signum).name
        logger.warning(f"Received {signal_name} (signal {signum}) - initiating graceful shutdown")
        self.kill_now = True
        self.signal_received = signum

    @classmethod
    def get_instance(cls) -> "GracefulKiller":
        """Get singleton instance - ensures only one handler is registered."""
        if cls._instance is None:
            cls._instance = GracefulKiller()
        return cls._instance

    def should_stop(self) -> bool:
        """Check if shutdown has been requested."""
        return self.kill_now

    def reset(self) -> None:
        """Reset the kill flag. Useful for testing."""
        self.kill_now = False
        self.signal_received = None
