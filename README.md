### RL control strategies for EVs fleet VPPs
# [Reinforcement Learning control strategies for Electric Vehicles fleet Virtual Power Plants]
Thesis based on the development of a RL agent that manages a VPP through EVs charging stations in an household environment. Main optimization objectives of the VPP are: Valley filling, peak shaving and zero resulting load over time. Main action performed to reach objectives are: storage of Renewable energy resources and power push in the grid at high demand times. 
The development of the Virtual Power Plant environment is based on the ELVIS (Electric Vehicles Infrastructure Simulator) open library from DAI-Labor:
https://github.com/dailab/elvis
![alt text](https://github.com/francescomaldonato/RL_VPP_Thesis/blob/main/data/images/Elvis_logo.png?raw=true)
The thesis code is currently available at:
https://github.com/francescomaldonato/RL_VPP_Thesis

## Outline:
This research has the intent to investigate on a sustainable way of life of a general household energy production and storage.
The main goal of the thesis is to explore the boundaries of a self-sustained energy system with minimum power coming from the grid and expenses. 
The energy production means are PV solar panel modules and domestic Wind turbines. The storage system is based on EVs batteries.
An RL agent will be in charge on managing EVs power resources to guarantee minimum charge left at EVs departure and optimizing peak shaving and valley filling of the power grid.
A scenario visualization of such implemented system is shown below.

![alt text](https://github.com/francescomaldonato/RL_VPP_Thesis/blob/main/data/images/RL_VPP_Thesis_scenario.png?raw=true)

A simulation configuration parameters set is shown below. [Assumptions of 20 EVs arrival per week for an average parking time of 24 hours in the grid]

![alt text](https://github.com/francescomaldonato/RL_VPP_Thesis/blob/main/data/images/Elvis_config.png?raw=true)

## Repository structure

RL_VPP_Thesis:
    - `VPP_environment.py` (Python script containing the environment definition and functions)
    - `VPP_simulator.ipynb` (Notebook to test the VPP performances and features with the best trained model, currently RecurrentPPO)
    - `RL_control-strategies_for_EVs_fleet_VPP.pdf` (Developed thesis paper of the research)

    Simulator_notebooks: (folder with other notebooks to test the VPP with different RL algorithm)
        - `A2C_VPP_simulator.ipynb`
        - `MaskablePPO_VPP_simulator.ipynb`
        - `TRPO_VPP_simulator.ipynb`
        - `RecurrentPPO_VPP_simulator.ipynb`
    
    Agent_trainer_notebooks: (folder with the notebooks to train the VPP RL agent with the indicated set of hyperparameters for each RL algorithm)
        - `A2C_VPP_agent_trainer.ipynb`
        - `MaskablePPO_VPP_agent_trainer.ipynb`
        - `TRPO_VPP_agent_trainer.ipynb`
        - `RecurrentPPO_VPP_agent_trainer.ipynb`
    
    Hyperparameters_sweep_notebooks: (folder with the notebooks to tune Hyperparameters of the VPP RL agents for each RL algorithm)
        - `A2C_VPP_Hyperp_Sweep.ipynb`
        - `MaskablePPO_VPP_Hyperp_Sweep.ipynb`
        - `TRPO_VPP_Hyperp_Sweep.ipynb`
        - `RecurrentPPO_VPP_Hyperp_Sweep.ipynb`
    
    trained_models: (folder with the trained models for each RL algorithm ready to be loaded)
        - A2C_models (folder)
        - MaskablePPO_models (folder)
        - TRPO_models (folder)
        - RecurrentPPO_models (folder)
    
    data:
        - `training_dataset_merger.ipynb` (notebook that visualizes and creates the training dataset table)
        - `testing_dataset_merger.ipynb` (notebook that visualizes and creates the testing dataset table)
        - `validating_dataset_merger.ipynb` (notebook that visualizes and creates the validating dataset table)

        config_builder: (folder containing the YAML simulation config files)
            - `wohnblock_household_simulation_adaptive.yaml`

        environment_optimized_output: (folder where to store the VPP optimized simulation data results)
        images: (folder with plots of the best results obtained)
        
        data_training: (folder with pre-processing notebooks, 2019 raw-data .csv files, and the created training dataset table)
        data_testing: (folder with pre-processing notebooks, 2020 raw-data .csv files, and the created testing dataset table)
        data_validating: (folder with pre-processing notebooks, 2018 raw-data .csv files, and the created validating dataset table)
    
    wandb: (folder with Weights&Biases training data stored)
        - tensorboard_log: (folder where training tensorboard log files are stored)

## Initialization (quick VPP simulation)
- Open the VPP_environment notebook on Google Colaboratory at:
    https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/VPP_simulator.ipynb
- Run the whole notebook

It will automatically clone in the remote machine the repository: https://github.com/francescomaldonato/RL_VPP_Thesis.git

## RL algorithm performance testing
- Open a VPP simulator notebook with a trained model loaded in the Simulator_notebooks folder on Google Colaboratory
- Run the whole notebook

# Input datasets visualization (training, testing, validating)
- Open a dataset-merger notebook in the data folder on Google Colaboratory
- Run the whole notebook

# Raw datasets pre-processing (training, testing, validating)
- Open a profile-creator notebook inside either the data_training(2019 data), data_testing(2020 data), data_validating(2022 data) folder on Google Colaboratory.
    Choose the profile-creator notebook among:
    - WT-Wind-energy-production                 (`create_wt_load_profile.ipynb`)
    - PV-Solar-energy-production                (`create_pv_load_profile.ipynb`)
    - Multiple-households-power-consumption     (`create_MultipleHousehold_load.ipynb`)
    - Single-households-power-consumption       (`create_household_load.ipynb`)
    - Electricity-market-prices                 (`create_electricity_prices.ipynb`)
- Run the whole notebook

### Weights&Biases account login
If you wish to train or to tune some algorithms (explained in the next sections) create a Wandb (Weights&Biases) account at https://wandb.ai/ to keep track of the experiments.
The Colab notebook will automatically sign-in and save experiments results in your account storage. If the notebook asks you to sign in at the wandb.login(relogin=True) command, follow the instructions in the cell (open your wandb access code page and copy-paste the code in the cell blank space).

## Model training
- Open the *_VPP_agent_trainer notebook for the (*)Algorithm you want to train, in the Agent_trainer_notebooks folder on Google Colaboratory
- [Optional] Select the Hyperparameters set in the run-config section
- Run the whole notebook

## Algorithm Hyperparameter tuning
- Open the *_VPP_agent_trainer notebook for the (*)Algorithm you want to train, in the Agent_trainer_notebooks folder on Google Colaboratory
- [Optional] Select the Hyperparameters sweep set in the sweep-config section
- Run the whole notebook
- Check Hyperparameter Sweep results at the experiment Sweep page.
![alt text](https://github.com/francescomaldonato/RL_VPP_Thesis/blob/main/data/images/Hyperp_sweep_results.png?raw=true)

## Authors
Francesco Maldonato
    - Personal contact:     francesco.maldonato97@gmail.com
    - DAI-Labor contact:    Francesco.Maldonato@dai-labor.de

# Acknowledgments
- Thesis supervisor:      M.Sc. Izgh Hadachi    (Izgh.hadachi@dai-labor.de)
- Graphic designer:       Marco Maldonato       (https://marcomaldonato.com/)
- Research institution:   DAI-Labor             (https://dai-labor.de/en/home/)
![alt text](https://github.com/francescomaldonato/RL_VPP_Thesis/blob/main/data/images/DAI_logo.png?raw=true)
