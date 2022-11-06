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

![alt text](https://github.com/francescomaldonato/RL_VPP_Thesis/blob/main/data/images/ELVIS_data_25.png?raw=true)
[Assumptions of 25 EVs arrival per week for an average parking time of 24 hours in the grid with an average of 50% available battery at arrival. Available car type: Tesla Model S]

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
- `MALDONATO-RL_control-strategies_for_EVs_fleet_VPP.pdf` (Developed thesis paper of the research) [Not available yet]

- Algorithm_simulator_notebooks: (folder with notebooks to test the VPP with different RL algorithms or with random actions)
    - `1-Random_VPP_simulator.ipynb`
    - `A2C_VPP_simulator.ipynb`
    - `MaskablePPO_VPP_simulator.ipynb`
    - `TRPO_VPP_simulator.ipynb`
    - `RecurrentPPO_VPP_simulator.ipynb`

- EV_experiment_notebooks: (folder with notebooks to test different EVs numbers (weekly arrivals) in the VPP simulation)
    - `EVs_RecurrentPPO_VPP_tester.ipynb`
    - `EVs_RecurrentPPO_VPP_validator.ipynb`
    - `35EVs_RecurrentPPO_VPP_simulator.ipynb`
    - `30EVs_RecurrentPPO_VPP_simulator.ipynb`
    - `25EVs_RecurrentPPO_VPP_simulator.ipynb`
    - `20EVs_RecurrentPPO_VPP_simulator.ipynb`
    - `15EVs_RecurrentPPO_VPP_simulator.ipynb`
    - `10EVs_RecurrentPPO_VPP_simulator.ipynb`

- Agent_trainer_notebooks: (folder with the notebooks to train the VPP RL agent with the indicated set of hyperparameters for each RL algorithm)
    - `A2C_VPP_agent_trainer.ipynb`
    - `MaskablePPO_VPP_agent_trainer.ipynb`
    - `TRPO_VPP_agent_trainer.ipynb`
    - `RecurrentPPO_VPP_agent_trainer.ipynb`
    
- Hyperparameters_sweep_notebooks: (folder with the notebooks to tune Hyperparameters of the VPP RL agents for each RL algorithm)
    - `A2C_VPP_Hyperp_Sweep.ipynb`
    - `MaskablePPO_VPP_Hyperp_Sweep.ipynb`
    - `TRPO_VPP_Hyperp_Sweep.ipynb`
    - `RecurrentPPO_VPP_Hyperp_Sweep.ipynb`
    
- trained_models: (folder with the trained models for each RL algorithm ready to be loaded)
    - A2C_models (folder)
    - MaskablePPO_models (folder)
    - TRPO_models (folder)
    - RecurrentPPO_models (folder)
    
- data:
    - `training_dataset_merger.ipynb` (notebook that visualizes and creates the training dataset table)
    - `testing_dataset_merger.ipynb` (notebook that visualizes and creates the testing dataset table)
    - `validating_dataset_merger.ipynb` (notebook that visualizes and creates the validating dataset table)
    - data_training: (folder with pre-processing notebooks, 2019 raw-data .csv files, and the created training dataset table)
    - data_testing: (folder with pre-processing notebooks, 2020 raw-data .csv files, and the created testing dataset table)
    - data_validating: (folder with pre-processing notebooks, 2018 raw-data .csv files, and the created validating dataset table)
    - config_builder: (folder containing the YAML simulation config files)
        - `wohnblock_household_simulation_adaptive.yaml`
        - `wohnblock_household_simulation_adaptive_30.yaml`
    - environment_optimized_output: (folder where to store the VPP optimized simulation data results)
        - `VPP_table.csv` (last VPP optimized simulation data results)
    - images: (folder with plots of the best results obtained)
    - algorithms_results: (folder with algorithm evaluation notebook and plots)
        - `Algorithms_results_plot.ipynb` (notebook that plots Algorithms performances)
        - algorithms_results_table: (folder containing algorithms sweep results tables downloaded from wandb.ai and VPP Experiments based on EVs arrivals)
        - algorithms_graphs: (folder containing algorithms results graphs)
    - wandb: (folder with Weights&Biases training data stored)
        - tensorboard_log: (folder where training tensorboard log files are stored)


## Different RL algorithms performance testing
- Open on Github a VPP-simulator notebook with a trained model loaded in the `Simulator_notebooks` folder. Choose among the algorithms:
    - A2C (Advantage Actor-Critic)
    - MaskablePPO (Maskable Proximal Policy Optimization)
    - TRPO (Trust Region Policy Optimization)
    - RecurrentPPO (Recurrent Proximal Policy Optimization)
- Load it on Google Colaboratory by substituting the URL address first part (https://github.com/) with the Colab github loader address (https://colab.research.google.com/github/). Direct access notebooks links:
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Algorithm_simulator_notebooks/A2C_VPP_simulator.ipynb
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Algorithm_simulator_notebooks/MaskablePPO_VPP_simulator.ipynb
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Algorithm_simulator_notebooks/TRPO_VPP_simulator.ipynb
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Algorithm_simulator_notebooks/RecurrentPPO_VPP_simulator.ipynb
- Check the VPP results and interactive plots already loaded in the notebook.
- If you wish to test again the VPP performances, re-run the whole notebook.

## VPP environment and Datasets debug with random-simulation
For debugging purposes and to cross-check datasets loading and Algorithm actual performances, the `1-Random_VPP_simulator.ipynb` notebook is provided to run random simulations without any RL model choosing actions. 
- Load it on Google Colaboratory by substituting the URL address first part (https://github.com/) with the Colab github loader address (https://colab.research.google.com/github/). Direct access notebooks links:
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Algorithm_simulator_notebooks/1-Random_VPP_simulator.ipynb

## Load different Elvis simulation config set and run experiments
In the `data/config_builder/` folder you can find the Elvis YAML config files.
- Create a new config file or modify the existing ones parameters to change the Vehicle arrival simulation characteristics. You can modify [Not possible ATM]: 
    - num_charging_events   (number of EVs arrival, weekly)
    - mean_park             (mean parking time, hours)
    - std_deviation_park    (standard deviation parking time, hours)
    - mean_soc              (mean State Of Charge of EVs at arrival, from 0 to 1)
    - std_deviation_soc     (standard deviation State Of Charge of EVs at arrival)
- Open the VPP simulation notebook you wish to test (as explained in previous section).
- In the "Load ELVIS YAML config file" section, load the config file you wish. Choose among the available config files by modifying the `case` string to:
    - `wohnblock_household_simulation_adaptive.yaml` (loaded by default, 20 EVs arrivals per week with 50% av.battery)
    - `wohnblock_household_simulation_adaptive_18.yaml` (18 EVs arrivals per week with 40% av.battery) 
    - `wohnblock_household_simulation_adaptive_22.yaml` (22 EVs arrivals per week with 55% av.battery) 
    - `wohnblock_household_simulation_adaptive_30.yaml` (30 EVs arrivals per week with 65% av.battery) 
- Then re-run the whole notebook to test the VPP experiment performances.

You can check the experiments results for different EVs numbers (weekly arrivals) already loaded in the folder `EV_experiment_notebooks`. Direct access notebooks links:
- https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/EV_experiment_notebooks/35EVs_RecurrentPPO_VPP_simulator.ipynb
- https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/EV_experiment_notebooks/30EVs_RecurrentPPO_VPP_simulator.ipynb
- https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/EV_experiment_notebooks/25EVs_RecurrentPPO_VPP_simulator.ipynb
- https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/EV_experiment_notebooks/20EVs_RecurrentPPO_VPP_simulator.ipynb
- https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/EV_experiment_notebooks/15EVs_RecurrentPPO_VPP_simulator.ipynb
- https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/EV_experiment_notebooks/10EVs_RecurrentPPO_VPP_simulator.ipynb

### Input datasets visualization (training, testing, validating)
- Open a dataset-merger notebook in the `data` folder, chosing among training, testing, validating.
- Load it on Google Colaboratory by substituting the URL address first part (https://github.com/) with the Colab github loader address (https://colab.research.google.com/github/). Direct access notebooks links:
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/training_dataset_merger.ipynb
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/testing_dataset_merger.ipynb
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/validating_dataset_merger.ipynb
- Check the dataset interactive plots already loaded in the notebook.
- If you wish to load again the dataset, re-run the whole notebook.

### Raw datasets pre-processing (training, testing, validating)
- Open a profile-creator notebook inside either the folder `data/data_training/`(2019 data), `data/data_testing/`(2020 data), `data/data_validating/`(2022 data).
    Choose the profile-creator notebook among:
    - WT-Wind-energy-production                 (`create_wt_load_profile.ipynb`)
    - PV-Solar-energy-production                (`create_pv_load_profile.ipynb`)
    - Multiple-households-power-consumption     (`create_MultipleHousehold_load.ipynb`)
    - Single-households-power-consumption       (`create_household_load.ipynb`, to check sigle households, not used in the dataset)
    - Electricity-market-prices                 (`create_electricity_prices.ipynb`)
- Load it on Google Colaboratoryby substituting the URL address first part (https://github.com/) with the Colab github loader address (https://colab.research.google.com/github/), or opening a direct access link provided below:
    - Training dataset direct access notebooks link:
        - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_training/create_wt_load_profile.ipynb
        - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_training/create_pv_load_profile.ipynb
        - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_training/create_MultipleHousehold_load.ipynb
        - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_training/create_electricity_prices.ipynb
    - Testing dataset direct access notebooks link:
        - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_testing/create_wt_load_profile.ipynb
        - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_testing/create_pv_load_profile.ipynb
        - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_testing/create_MultipleHousehold_load.ipynb
        - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_testing/create_electricity_prices.ipynb
    - Validating dataset direct access notebooks link:
        - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_validating/create_wt_load_profile.ipynb
        - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_validating/create_pv_load_profile.ipynb
        - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_validating/create_MultipleHousehold_load.ipynb
        - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/data_validating/create_electricity_prices.ipynb
- Check the profile interactive plots already loaded in the notebook.
- If you wish to load again the profile data, re-run the whole notebook.

### Weights&Biases account login
If you wish to train or to tune some algorithms (explained in the next sections) create a Wandb (Weights&Biases) account at https://wandb.ai/ to keep track of the experiments.
The Colab notebook will automatically sign-in and save experiments results in your account storage. If the notebook asks you to sign in at the wandb.login(relogin=True) command, follow the instructions in the cell (open your wandb access code page and copy-paste the code in the cell blank space).

## Model training
You can train your own model with your Hyperparameters set.
- In the `Agent_trainer_notebooks` folder, open the (X)_VPP_agent_trainer notebook for the (X) model Algorithm you want to train. Choose among the algorithms:
    - A2C (Advantage Actor-Critic)
    - MaskablePPO (Maskable Proximal Policy Optimization)
    - TRPO (Trust Region Policy Optimization)
    - RecurrentPPO (Recurrent Proximal Policy Optimization)
- Load it on Google Colaboratory by substituting the URL address first part (https://github.com/) with the Colab github loader address (https://colab.research.google.com/github/). Direct access notebooks links:
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Agent_trainer_notebooks/A2C_VPP_agent_trainer.ipynb
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Agent_trainer_notebooks/MaskablePPO_VPP_agent_trainer.ipynb
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Agent_trainer_notebooks/TRPO_VPP_agent_trainer.ipynb
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Agent_trainer_notebooks/RecurrentPPO_VPP_agent_trainer.ipynb
- [Recommended] Switch the Runtime type to "GPU Accellerator"
- [Optional] Select the Hyperparameters set in the run-config section. The Algorithms class and parameters documentation is available at:
    - https://stable-baselines3.readthedocs.io/en/master/index.html
    - https://stable-baselines3.readthedocs.io/en/master/guide/sb3_contrib.html
- Run the notebook to start training the model.

## Algorithm Hyperparameter tuning
You can launch an Hyperparameters sweep session for a selected algorithm.
- In the `Hyperparameters_sweep_notebooks` folder, open the (X)_VPP_agent_trainer notebook for the (X) Algorithm you want to tune. Choose among the algorithms:
    - A2C (Advantage Actor-Critic)
    - MaskablePPO (Maskable Proximal Policy Optimization)
    - TRPO (Trust Region Policy Optimization)
    - RecurrentPPO (Recurrent Proximal Policy Optimization)
- Load it on Google Colaboratory by substituting the URL address first part (https://github.com/) with the Colab github loader address (https://colab.research.google.com/github/). Direct access notebooks links:
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Hyperparameters_sweep_notebooks/A2C_VPP_Hyperp_Sweep.ipynb
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Hyperparameters_sweep_notebooks/MaskablePPO_VPP_Hyperp_Sweep.ipynb
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Hyperparameters_sweep_notebooks/TRPO_VPP_Hyperp_Sweep.ipynb
    - https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/Hyperparameters_sweep_notebooks/RecurrentPPO_VPP_Hyperp_Sweep.ipynb
- [Recommended] Switch the Runtime type to "GPU Accellerator"
- [Optional] Select the Hyperparameters sweep set in the sweep-config section. Weights and Biases Sweep tutorial: https://docs.wandb.ai/guides/sweeps
- Run the whole notebook to start tuning the model
- Check Hyperparameter Sweep results at the experiment Sweep page.

## Algorithm and Experiments results graphs
Plot the Hyperparameters sweep results and the algorithm performances obtained and stored in the `data/algorithms_results/algorithms_results_table` in the notebook:
- https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/algorithms_results/Algorithms_results_plots.ipynb

Plot the VPP tuning experiments results (for testing and validating datatsets) based on EVs arrival, from the `data/algorithms_results/algorithms_results_table` in the notebook:
- https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/algorithms_results/Experiments_testing_plots.ipynb
- https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/data/algorithms_results/Experiments_validator_plots.ipynb

The tables are extracted from the wandb.ai Sweep page for each Algorithm. Check out the 2D and 3D graphs already loaded.

![alt text](https://github.com/francescomaldonato/RL_VPP_Thesis/blob/main/data/images/3D_bubble_names_colors.png?raw=true)

## Enjoy the material!
Have fun training, tuning and testing the RL algorithms with interactive graphs while understanding how a Virtual Power Plant works.

## Authors
Francesco Maldonato
    - Personal contact:     francesco.maldonato97@gmail.com
    - DAI-Labor contact:    Francesco.Maldonato@dai-labor.de

## Acknowledgments
- Thesis supervisor:      M.Sc. Izgh Hadachi    (Izgh.hadachi@dai-labor.de)
- Graphic designer:       Marco Maldonato       (https://marcomaldonato.com/)
- Research institution:   DAI-Labor             (https://dai-labor.de/en/home/)
![alt text](https://github.com/francescomaldonato/RL_VPP_Thesis/blob/main/data/images/DAI_logo.png?raw=true)
