"""Control modules: PID/Fuzzy/FEAR-PID, mixer, and DDQN gain adjuster."""
from framework.control.pid import PIDController, PIDGains
from framework.control.fuzzy_pid import FuzzyPIDController, FuzzyConfig
from framework.control.fear_pid import FEARPIDController, FEARConfig
from framework.control.mixer import MixerConfig, mix_thrust_torque
from framework.control.ddqn import DDQNGainAdjuster, train_ddqn_gain_adjuster, DDQNConfig

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
