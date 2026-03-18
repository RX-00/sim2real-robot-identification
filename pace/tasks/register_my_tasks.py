import gymnasium as gym

gym.register(
    id="IsaacLab-Pace-Go2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "go2_pace_env_cfg:Go2PaceEnvCfg",
    },
)