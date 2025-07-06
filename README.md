# harmony-TDMPC2: Precision Rocket Landing with Advanced Reinforcement Learning

This repository showcases the power of **harmony-TDMPC2**, a novel reinforcement learning algorithm, applied to the critical challenge of autonomous rocket landing.

## Project Highlights

We've trained an agent using **harmony-TDMPC2** to achieve safe and efficient rocket landings. This project demonstrates the algorithm's capability in mastering complex dynamics and optimizing control policies within simulated rocket landing environments.

## Visual Proof of Success

Witness the **harmony-TDMPC2** agent's successful rocket landings in action:

<p align="center">
  <img src="https://github.com/PaoShou-China/harmony-TDMPC2/blob/main/video/1.gif" width="30%" alt="Rocket Landing Demo 1"/>
  <img src="https://github.com/PaoShou-China/harmony-TDMPC2/blob/main/video/2.gif" width="30%" alt="Rocket Landing Demo 2"/>
  <img src="https://github.com/PaoShou-China/harmony-TDMPC2/blob/main/video/3.gif" width="30%" alt="Rocket Landing Demo 3"/>
</p>

## Core Features

*   **Advanced Reinforcement Learning:** Utilizes cutting-edge RL techniques for optimal control.
*   **harmony-TDMPC2 Algorithm:** Implements a state-of-the-art TDMPC2 with unique harmony-inspired enhancements.
*   **Rocket Landing Focus:** Specifically engineered for the intricate control of rocket trajectories and precise landings.
*   **Simulation-Driven:** Developed and rigorously tested in a high-fidelity simulation environment.

## Quick Start Guide

Get your environment set up and run the project with these simple steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/PaoShou-China/harmony-TDMPC2.git
    cd harmony-TDMPC2
    ```

2.  **Set Up Conda Environment & Dependencies:**
    All project dependencies, including `tdmpc2`, are managed via `environment.yaml`.

    *   **Create and Activate Environment:**
        ```bash
        conda env create -f environment.yaml
        conda activate tdmpc2
        ```
        *(Note: The environment name `tdmpc2` is assumed from `environment.yaml`. Adjust if your file specifies a different name.)*

    *   **Integrate Custom PyFlyt Version:**
        This project requires a specific version of PyFlyt, available at [https://github.com/defrag-bambino/MyPyFlyt/tree/master](https://github.com/defrag-bambino/MyPyFlyt/tree/master).
        After activating your Conda environment, please manually replace the `[PyFlyt/gym_envs/rocket_envs/rocket_landing_env.py]` files in your project directory with the corresponding files from the `MyPyFlyt` repository. Refer to the `MyPyFlyt` repository for detailed integration instructions.

3.  **Run the Simulation:**
    With your environment activated, execute the training script:
    ```bash
    python train.py
    ```

## Contribute

We welcome your contributions! Feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
