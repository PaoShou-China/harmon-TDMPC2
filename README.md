# harmony-TDMPC2: Rocket Landing with Harmony-TDMPC2

This repository demonstrates the application of the **harmony-TDMPC2** algorithm, a novel approach in reinforcement learning, to the challenging problem of rocket landing.

## Project Overview

We utilize harmony-TDMPC2 to train an agent capable of autonomously landing a rocket safely and efficiently. The algorithm's ability to handle complex dynamics and optimize control policies is showcased through simulated rocket landing scenarios.

## Visual Demonstration

The following GIFs illustrate the successful rocket landing process achieved by the harmony-TDMPC2 agent:

<p align="center">
  <img src="https://github.com/PaoShou-China/harmony-TDMPC2/blob/main/video/1.gif" width="30%"/>
  <img src="https://github.com/PaoShou-China/harmony-TDMPC2/blob/main/video/2.gif" width="30%"/>
  <img src="https://github.com/PaoShou-China/harmony-TDMPC2/blob/main/video/3.gif" width="30%"/>
</p>

## Key Features

*   **Reinforcement Learning:** Leverages advanced RL techniques for optimal control.
*   **harmony-TDMPC2:** Implements a state-of-the-art TDMPC2 algorithm with harmony-inspired enhancements.
*   **Rocket Landing:** Specifically targets the complex dynamics of rocket trajectory control and landing.
*   **Simulation-based:** Developed and tested within a simulated environment.

## Getting Started

To set up the environment and run the project, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PaoShou-China/harmony-TDMPC2.git
    cd harmony-TDMPC2
    ```

2.  **Create and activate a Conda environment:**
    This project uses a `environment.yaml` file to manage all dependencies, including `tdmpc2` and `PyFlyt`.
    ```bash
    conda env create -f environment.yaml
    conda activate tdmpc2 # Replace
    ```
    *(Ensure that your `environment.yaml` file correctly specifies the installation of `tdmpc2` and `PyFlyt`, either by including them directly if they are available as conda packages, or by using `pip` within the YAML to install them from their respective GitHub repositories.)*

3.  **Run the simulation:**
    Once the environment is activated, you can run the project's code directly.
    ```bash
    python train.py
    ```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
