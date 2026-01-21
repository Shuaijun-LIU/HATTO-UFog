"""Control modules: PID/Fuzzy/FEAR-PID, mixer, and DDQN gain adjuster."""
from ufog_network.control.pid import PIDController, PIDGains
from ufog_network.control.fuzzy_pid import FuzzyPIDController, FuzzyConfig
from ufog_network.control.fear_pid import FEARPIDController, FEARConfig
from ufog_network.control.mixer import MixerConfig, mix_thrust_torque
try:
    from ufog_network.control.ddqn import DDQNGainAdjuster, train_ddqn_gain_adjuster, DDQNConfig
    _DDQN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _DDQN_AVAILABLE = False

    def _ddqn_missing(*_args, **_kwargs):
        raise ImportError("DDQN gain adjuster requires torch; install torch to enable.")

    class DDQNConfig:  # type: ignore[no-redef]
        pass

    class DDQNGainAdjuster:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs) -> None:
            _ddqn_missing()

    def train_ddqn_gain_adjuster(*_args, **_kwargs):  # type: ignore[no-redef]
        _ddqn_missing()

__all__ = [
    "PIDController",
    "PIDGains",
    "FuzzyPIDController",
    "FuzzyConfig",
    "FEARPIDController",
    "FEARConfig",
    "MixerConfig",
    "mix_thrust_torque",
    "DDQNGainAdjuster",
    "train_ddqn_gain_adjuster",
    "DDQNConfig",
]
