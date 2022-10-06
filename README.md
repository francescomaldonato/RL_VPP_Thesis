# RL control strategies for EVs fleet VPPs
### [Reinforcement Learning control strategies for Electric Vehicles fleet Virtual Power Plants]
Thesis based on the development of a RL agent that manages a VPP through EVs charging stations in an household environment. Main optimization objectives of the VPP are: Valley filling, peak shaving and zero resulting load over time. Main action performed to reach objectives are: storage of Renewable energy resources and power push in the grid at high demand times. The development of the Virtual Power Plant environment is based on the ELVIS (Electric Vehicles Infrastructure Simulator) open library from DAI-Labor: https://github.com/dailab/elvis
The thesis code is currently available at: (https://github.com/francescomaldonato/RL_VPP_Thesis)

## Outline:
This research has the intent to investigate on a sustainable way of life of a general household energy production and storage.
The main goal of the thesis is to explore the boundaries of a self-sustained energy system with minimum power coming from the grid and expenses. 
The energy production means are PV solar panel modules and domestic Wind turbines. The storage system is based on EVs batteries.
An RL agent will be in charge on managing EVs power resources to guarantee minimum charge left at EVs departure and optimizing peak shaving and valley filling of the power grid.
A scenario visualization of such implemented system is shown below.

![alt text](https://github.com/francescomaldonato/RL_VPP_Thesis/blob/main/data/images/RL_VPP_Thesis_scenario.png?raw=true)

The simulation configuration parameters set already loaded is shown below. This can be changed by modifying the ELvis config file in the `data/config_builder/` folder (explained below).

![alt text](https://github.com/francescomaldonato/RL_VPP_Thesis/blob/main/data/images/Elvis_config.png?raw=true)
[Assumptions of 20 EVs arrival per week for an average parking time of 24 hours in the grid with an average of 50% available battery at arrival. Available car type: Tesla Model S]

## Initialization (quick VPP simulation)
- Open the VPP_environment notebook on Google Colaboratory at:
    https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/VPP_simulator.ipynb
- Check the VPP results and interactive plots already loaded in the notebook.
- If you wish to test again the VPP performances, re-run the whole notebook.

It will automatically clone in the remote machine the repository: https://github.com/francescomaldonato/RL_VPP_Thesis.git

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
            - `wohnblock_household_simulation_adaptive_30.yaml`

        environment_optimized_output: (folder where to store the VPP optimized simulation data results)
        images: (folder with plots of the best results obtained)
        
        data_training: (folder with pre-processing notebooks, 2019 raw-data .csv files, and the created training dataset table)
        data_testing: (folder with pre-processing notebooks, 2020 raw-data .csv files, and the created testing dataset table)
        data_validating: (folder with pre-processing notebooks, 2018 raw-data .csv files, and the created validating dataset table)
    
    wandb: (folder with Weights&Biases training data stored)
        - tensorboard_log: (folder where training tensorboard log files are stored)


## Different RL algorithms performance testing
- Open on Github a VPP-simulator notebook with a trained model loaded in the `Simulator_notebooks` folder. Choose among the algorithms:
    - A2C (Asynchronous Actor Critic)
    - MaskablePPO (Maskable Proximal Policy Optimization)
    - TRPO (Trust Region Policy Optimization)
    - RecurrentPPO (Recurrent Proximal Policy Optimization)
- Load it on Google Colaboratory by substituting the URL address first part (https://github.com/) with the Colab github loader address (https://colab.research.google.com/github/). E.g:
    https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Simulator_notebooks/TRPO_VPP_simulator.ipynb
- Check the VPP results and interactive plots already loaded in the notebook.
- If you wish to test again the VPP performances, re-run the whole notebook.

## Load different Elvis simulation config set and run experiments
In the `data/config_builder/` folder you can find the Elvis YAML config files.
- Create a new config file or modify the existing ones parameters to change the Vehicle arrival simulation characteristics. You can modify: 
    - num_charging_events   (number of EVs arrival, weekly)
    - mean_park             (mean parking time, hours)
    - std_deviation_park    (standard deviation parking time, hours)
    - mean_soc              (mean State Of Charge of EVs at arrival, from 0 to 1)
    - std_deviation_soc     (standard deviation State Of Charge of EVs at arrival)
- Open the VPP simulation notebook you wish to test (as explained in previous section).
- In the "Load ELVIS YAML config file" section, load the config file you wish.
- Then re-run the whole notebook to test the VPP experiment performances.

### Input datasets visualization (training, testing, validating)
- Open a dataset-merger notebook in the `data` folder, chosing among training, testing, validating.
- Load it on Google Colaboratory by substituting the URL address first part (https://github.com/) with the Colab github loader address (https://colab.research.google.com/github/). E.g:
    https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/testing_dataset_merger.ipynb
- Check the dataset interactive plots already loaded in the notebook.
- If you wish to load again the dataset, re-run the whole notebook.

### Raw datasets pre-processing (training, testing, validating)
- Open a profile-creator notebook inside either the folder `data/data_training/`(2019 data), `data/data_testing/`(2020 data), `data/data_validating/`(2022 data).
    Choose the profile-creator notebook among:
    - WT-Wind-energy-production                 (`create_wt_load_profile.ipynb`)
    - PV-Solar-energy-production                (`create_pv_load_profile.ipynb`)
    - Multiple-households-power-consumption     (`create_MultipleHousehold_load.ipynb`)
    - Single-households-power-consumption       (`create_household_load.ipynb`)
    - Electricity-market-prices                 (`create_electricity_prices.ipynb`)
- Load it on Google Colaboratoryby substituting the URL address first part (https://github.com/) with the Colab github loader address (https://colab.research.google.com/github/). E.g:
    https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_training/create_MultipleHousehold_load.ipynb
- Check the profile interactive plots already loaded in the notebook.
- If you wish to load again the profile data, re-run the whole notebook.

### Weights&Biases account login
If you wish to train or to tune some algorithms (explained in the next sections) create a Wandb (Weights&Biases) account at https://wandb.ai/ to keep track of the experiments.
The Colab notebook will automatically sign-in and save experiments results in your account storage. If the notebook asks you to sign in at the wandb.login(relogin=True) command, follow the instructions in the cell (open your wandb access code page and copy-paste the code in the cell blank space).

## Model training
You can train your own model with your Hyperparameters set.
- In the `Agent_trainer_notebooks` folder, open the (X)_VPP_agent_trainer notebook for the (X) model Algorithm you want to train. Choose among the algorithms:
    - A2C (Asynchronous Actor Critic)
    - MaskablePPO (Maskable Proximal Policy Optimization)
    - TRPO (Trust Region Policy Optimization)
    - RecurrentPPO (Recurrent Proximal Policy Optimization)
- Load it on Google Colaboratory by substituting the URL address first part (https://github.com/) with the Colab github loader address (https://colab.research.google.com/github/). E.g:
    https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Agent_trainer_notebooks/MaskablePPO_VPP_agent_trainer.ipynb
- [Recommended] Switch the Runtime type to "GPU Accellerator"
- [Optional] Select the Hyperparameters set in the run-config section. The Algorithms class and parameters documentation is available at:
    - https://stable-baselines3.readthedocs.io/en/master/index.html
    - https://stable-baselines3.readthedocs.io/en/master/guide/sb3_contrib.html
- Run the notebook to start training the model.

## Algorithm Hyperparameter tuning
You can launch an Hyperparameters sweep session for a selected algorithm.
- In the `Agent_trainer_notebooks` folder, open the (X)_VPP_agent_trainer notebook for the (X) Algorithm you want to tune. Choose among the algorithms:
    - A2C (Asynchronous Actor Critic)
    - MaskablePPO (Maskable Proximal Policy Optimization)
    - TRPO (Trust Region Policy Optimization)
    - RecurrentPPO (Recurrent Proximal Policy Optimization)
- Load it on Google Colaboratory by substituting the URL address first part (https://github.com/) with the Colab github loader address (https://colab.research.google.com/github/). E.g:
    https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Hyperparameters_sweep_notebooks/A2C_VPP_Hyperp_Sweep.ipynb
- [Recommended] Switch the Runtime type to "GPU Accellerator"
- [Optional] Select the Hyperparameters sweep set in the sweep-config section. Weights and Biases Sweep tutorial: https://docs.wandb.ai/guides/sweeps
- Run the whole notebook to start tuning the model
- Check Hyperparameter Sweep results at the experiment Sweep page.
![alt text](https://github.com/francescomaldonato/RL_VPP_Thesis/blob/main/data/images/Hyperp_sweep_results.png?raw=true)

## Authors
Francesco Maldonato
    - Personal contact:     francesco.maldonato97@gmail.com
    - DAI-Labor contact:    Francesco.Maldonato@dai-labor.de

## Acknowledgments
- Thesis supervisor:      M.Sc. Izgh Hadachi    (Izgh.hadachi@dai-labor.de)
- Graphic designer:       Marco Maldonato       (https://marcomaldonato.com/)
- Research institution:   DAI-Labor             (https://dai-labor.de/en/home/)
![alt text](https://github.com/francescomaldonato/RL_VPP_Thesis/blob/main/data/images/DAI_logo.png?raw=true)
