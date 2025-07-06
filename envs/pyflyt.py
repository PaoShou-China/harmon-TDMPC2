import numpy as np
import gymnasium as gym
from collections import defaultdict
import torch
import os
import importlib.util
from typing import Any, Dict, Optional, Tuple
import yaml
from yaml import CDumper # CDumper is a faster C implementation of Dumper
import PyFlyt.gym_envs # Required for gym.make("PyFlyt/Rocket-Landing-v4")

# Assuming envs.wrappers.timeout.Timeout is a custom wrapper available in your project.
# If not, you might need to define it or use gym.wrappers.TimeLimit.
from envs.wrappers.timeout import Timeout


# Define the rocket parameters template based on the provided tables.
# Each parameter can have:
# - "fix": A fixed value (used in both 'random' and 'nominal' modes if no 'min'/'max' is present).
# - "min", "max": A range for uniform random sampling (used in 'random' mode).
# - "nominal": The default or desired fixed value (used in 'nominal' mode).
rocket = {
    "booster_params": {
        # Parameters from "Randomized Parameters for the Rocket Booster Engines (Per Engine)" (Table 3)
        # and "Fixed Global Parameters for the Rocket Landing Simulation Environment" (Table 1)
        "total_fuel": {"min": 369.81, 'max': 451.99, "nominal": 410.9},            # Initial Fuel Mass (kg)
        "max_fuel_rate": {"min": 1.3059, "max": 1.5961, "nominal": 1.451},         # Maximum Fuel Rate (kg/s)
        "inertia_ixx": {"min": 1510.2, "max": 1845.8, "nominal": 1678},           # Moment of inertia (X-axis) (kg·m²)
        "inertia_iyy": {"min": 1510.2, "max": 1845.8, "nominal": 1678},           # Moment of inertia (Y-axis) (kg·m²)
        "inertia_izz": {"min": 6.309, "max": 7.711, "nominal": 7.01},           # Moment of inertia (Z-axis) (kg·m²)
        "min_thrust": {"min": 2670.03, "max": 3263.37, "nominal": 2966.7},          # Minimum Thrust (N)
        "max_thrust": {"fix": 7607.0, "nominal": 7607.0},          # Maximum Total Thrust (kN) - Fixed as per Table 1
        "reignitable": {"fix": True, "nominal": True},           # Engine restart capability flag (not in tables, kept fixed)
        "gimbal_range_degrees": {"fix": 5.0, "nominal": 5.0},     # Gimbal Range (°) - Randomized as per Table 3
        "booster_tau": {"min": 0.009, "max": 0.011, "nominal": 0.01},           # Booster Actuator Time Constant (s)
        "gimbal_tau": {"min": 0.009, "max": 0.011, "nominal": 0.01},            # Gimbal Actuator Time Constant (s)
        "noise_ratio": {"fix": 0.01, "nominal": 0.01},           # Control signal noise injection ratio (not in tables, kept fixed)
    },
    "finlet_params": {
        # Parameters from "Randomized Aerodynamic and Actuator Parameters for the Rocket Finlets" (Table 4)
        "Cl_alpha_2D": {"min": 5.6547, "max": 6.9113, "nominal": 6.283},          # 2D Lift Curve Slope
        "chord": {"min": 0.45, "max": 0.55, "nominal": 0.5},                  # Finlet Chord (m)
        "span": {"min": 0.45, "max": 0.55, "nominal": 0.5},                   # Finlet Span (m)
        "flap_to_chord": {"fix": 1.0,  "nominal": 1.0},          # Flap to Chord Ratio
        "eta": {"min": 0.585, "max": 0.715, "nominal": 0.65},                   # Viscosity Correction Factor
        "alpha_0_base": {"fix": 0.0, "nominal": 0.0},          # Zero-lift angle of attack (°) (not in tables, kept fixed)
        "alpha_stall_P_base": {"min": 18.0, "max": 22.0, "nominal": 20.0},   # Positive Stall Angle of Attack (°)
        "alpha_stall_N_base": {"min": -22.0, "max": -18.0, "nominal": -20.0},  # Negative Stall Angle of Attack (°)
        "Cd_0": {"min": 0.009, "max": 0.011, "nominal": 0.01},                 # Zero-Lift Drag Coefficient
        "deflection_limit": {"min": 40.5, "max": 49.5, "nominal": 45.0},     # Deflection Limit (°)
        "tau": {"min": 0.045, "max": 0.055, "nominal": 0.05},                   # Actuator Time Constant (s)
    },
    "body_params": {
        # Parameters from "Randomized Aerodynamic Parameters for the Rocket Body" (Table 2)
        "drag_coef_x": {"min": 1.044, "max": 1.276, "nominal": 1.16},           # Drag Coefficient X
        "drag_coef_y": {"min": 1.044, "max": 1.276, "nominal": 1.16},           # Drag Coefficient Y
        "drag_coef_z": {"min": 1.8, "max": 2.2, "nominal": 2.0},            # Drag Coefficient Z
        "area_x": {"min": 1.58841, "max": 1.94139, "nominal": 1.7649},              # Reference Area X (m²)
        "area_y": {"min": 1.58841, "max": 1.94139, "nominal": 1.7649},              # Reference Area Y (m²)
        "area_z": {"min": 0.09675, "max": 0.11825, "nominal": 0.1075},              # Reference Area Z (m²)
    }
}


class UnifiedPyFlytWrapper(gym.Wrapper):
    """
    A unified wrapper for PyFlyt environments, handling:
    1. Action mapping from [-1, 1] to the environment's original action space.
    2. Observation type conversion to np.float32.
    3. Info dictionary standardization (adding 'success', 'terminated' keys).
    4. Providing a `rand_act` method for sampling actions as torch tensors.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Ensure the wrapped environment has a Box action space for mapping
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError(
                f"UnifiedPyFlytWrapper only supports Box action spaces, "
                f"but the wrapped environment has action space of type {type(env.action_space)}"
            )

        # Store the original environment's action space range for mapping
        self._min_action = self.env.action_space.low
        self._max_action = self.env.action_space.high

        # Set the wrapper's action space to [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.env.action_space.shape,
            dtype=np.float32 # Ensure action space dtype is float32 for consistency
        )

        # Set the wrapper's observation space to float32 and unbounded
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=env.observation_space.shape, dtype=np.float32
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the environment and converts the observation to float32.
        """
        obs, info = self.env.reset(**kwargs)
        return obs.astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float32, bool, Dict[str, Any]]:
        """
        Maps the input action from [-1, 1] to the original environment's action space range,
        performs the step, converts observation and reward types, and standardizes info.
        """
        # 1. Action mapping from [-1, 1] to original env's range
        action = np.clip(action, -1.0, 1.0)
        scaled_action = self._min_action + (action + 1.0) * (self._max_action - self._min_action) / 2.0

        # Perform the step in the wrapped environment
        obs, reward, terminated, truncated, info = self.env.step(scaled_action)

        # 2. Observation type conversion
        obs = obs.astype(np.float32)

        # 3. Info dictionary standardization
        info = defaultdict(float, info)
        info['success'] = float(info.get('env_complete', False))
        info['terminated'] = float(terminated)

        # Determine if the episode is done
        done = terminated or truncated

        return obs, np.float32(reward), done, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def rand_act(self) -> torch.Tensor:
        """
        Samples a random action from the wrapper's action space ([-1, 1])
        and returns it as a torch.Tensor.
        """
        return torch.from_numpy(self.action_space.sample().astype(np.float32))


class RandomizedVehicleWrapper(gym.Wrapper):
    """
    A wrapper that randomizes or sets nominal parameters for any vehicle in PyFlyt
    and writes them to the corresponding YAML config file.

    It supports two modes for parameter selection:
    1. 'random': Parameters are sampled uniformly within 'min'/'max' ranges,
                 or use 'fix' values if only 'fix' is provided.
    2. 'nominal': Parameters are set to their predefined 'nominal' values.

    The wrapper automatically locates the vehicle's configuration file based on
    the `vehicle_type` (e.g., 'rocket', 'quadx').
    """

    def __init__(
        self,
        env: gym.Env,
        vehicle_type: str,
        param_template: Dict[str, Any],
        mode: str = 'random',  # 'random' or 'nominal'
        seed: Optional[int] = None
    ):
        """
        Initializes the RandomizedVehicleWrapper.

        Args:
            env: The Gymnasium environment to wrap.
            vehicle_type: The name of the vehicle (e.g., 'rocket', 'quadx') used to locate its YAML config file.
            param_template: A nested dictionary defining the parameters.
                            Each leaf node (parameter definition) must contain:
                                - {"min": a, "max": b, "nominal": c} for range-based parameters.
                                - {"fix": x, "nominal": x} for fixed parameters.
                            The 'nominal' key is crucial for 'nominal' mode and identifies a parameter.
            mode: The mode for parameter selection. Must be either 'random' or 'nominal'.
                  Defaults to 'random'.
            seed: An optional integer seed for the random number generator, ensuring
                  deterministic parameter sampling in 'random' mode.
        """
        super().__init__(env)

        # Validate the specified mode
        if mode not in ['random', 'nominal']:
            raise ValueError(f"Invalid mode '{mode}'. Mode must be 'random' or 'nominal'.")
        self.mode = mode

        # Locate the root installation path of PyFlyt to find vehicle YAML configs
        pyflyt_spec = importlib.util.find_spec("PyFlyt")
        if pyflyt_spec is None or pyflyt_spec.origin is None:
            raise RuntimeError("Could not find PyFlyt installation path. Please ensure PyFlyt is installed.")

        pyflyt_root = os.path.dirname(pyflyt_spec.origin)
        self.vehicle_yaml_path = os.path.join(
            pyflyt_root, "models", "vehicles", vehicle_type, f"{vehicle_type}.yaml"
        )

        # Verify that the target YAML configuration file exists
        if not os.path.isfile(self.vehicle_yaml_path):
            raise FileNotFoundError(f"YAML config not found for vehicle type '{vehicle_type}' at: {self.vehicle_yaml_path}")

        self.param_template = param_template
        self.rng = np.random.default_rng(seed)

    def _get_nominal_parameters_recursive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively retrieves nominal parameter values from the nested dictionary.
        This function is used when `self.mode` is 'nominal'.
        """
        nominal_params = {}
        for key, val in data.items():
            if not isinstance(val, dict):
                raise ValueError(f"Unexpected non-dict value found at key '{key}': {val}. Expected a dictionary for parameter definition.")

            # If 'nominal' key is present, it's a leaf node (parameter definition)
            if 'nominal' in val:
                nominal_params[key] = val['nominal']
            else:
                # Not a leaf node, recurse into the sub-dictionary
                nominal_params[key] = self._get_nominal_parameters_recursive(val)
        return nominal_params

    def _get_random_parameters_recursive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively samples random parameter values from the nested dictionary.
        This function is used when `self.mode` is 'random'.
        """
        random_params = {}
        for key, val in data.items():
            if not isinstance(val, dict):
                raise ValueError(f"Unexpected non-dict value found at key '{key}': {val}. Expected a dictionary for parameter definition.")

            # If 'nominal' key is present, it's a leaf node (parameter definition)
            if 'nominal' in val:
                if 'min' in val and 'max' in val:
                    random_params[key] = self.rng.uniform(low=val['min'], high=val['max'])
                elif 'fix' in val:
                    random_params[key] = val['fix']
                else:
                    raise ValueError(f"Parameter '{key}' has 'nominal' but lacks 'min'/'max' or 'fix' for 'random' mode.")
            else:
                # Not a leaf node, recurse into the sub-dictionary
                random_params[key] = self._get_random_parameters_recursive(val)
        return random_params

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """
        Resets the environment. Before resetting the wrapped environment, it generates
        new parameters (either randomized or nominal based on `self.mode`) and
        writes them to the vehicle's YAML configuration file.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Determine the mode and call the appropriate parameter retrieval method
        if self.mode == 'nominal':
            selected_parameters = self._get_nominal_parameters_recursive(self.param_template)
        elif self.mode == 'random':
            selected_parameters = self._get_random_parameters_recursive(self.param_template)
        else:
            raise RuntimeError(f"Unknown mode '{self.mode}'. This should have been caught during initialization.")

        # Write the selected parameters to the vehicle's YAML configuration file
        try:
            with open(self.vehicle_yaml_path, 'w') as f:
                yaml.dump(
                    selected_parameters,
                    f,
                    default_flow_style=False, # Use block style for readability
                    sort_keys=False,          # Preserve original key order if possible
                    Dumper=CDumper            # Use faster C Dumper if available
                )
        except Exception as e:
            raise RuntimeError(f"Failed to write to YAML config at: {self.vehicle_yaml_path}. Error: {e}") from e

        # Call the wrapped environment's reset method
        return super().reset(seed=seed, options=options)


def make_env(args):
    """
    Create a PyFlyt Rocket Landing environment and wrap it with RandomizedVehicleWrapper
    and UnifiedPyFlytWrapper.
    The environment ID is fixed to "PyFlyt/Rocket-Landing-v4".

    Args:
        args: An object containing configuration attributes,
              e.g., args.seed (int) and args.mode (str).

    Returns:
        gym.Env: Wrapped Gymnasium environment instance.
    """
    # Create the base PyFlyt environment
    env = gym.make("PyFlyt/Rocket-Landing-v4")
    max_episode_steps = env.unwrapped.max_steps

    # Apply RandomizedVehicleWrapper first to set vehicle parameters
    env = RandomizedVehicleWrapper(
        env=env,
        vehicle_type="rocket",
        param_template=rocket,
        mode=args.mode, # Use args.mode from the input object
        seed=args.seed  # Use args.seed from the input object
    )


    # Apply the unified wrapper for action mapping, observation/info processing
    env = UnifiedPyFlytWrapper(env)

    # Add Gymnasium's RecordEpisodeStatistics wrapper for episode return and length tracking
    env.max_episode_steps = max_episode_steps

    return env

