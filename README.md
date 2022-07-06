### RL control strategies for EVs fleet VPPs
# [Reinforcement Learning control strategies for Electric Vehicles fleet Virtual Power Plants]
Thesis based on the development of a RL agent that manages a VPP through EVs charging stations. Main optimization objectives of the VPP are: Valley filling and peak shaving. Main action performed to reach objectives are: storage of Renewable energy resources and power push in the grid at high demand times. Assumptions of high number of vehicles connected for minimum time of 3-4 hours in the grid.

The thesis code is currently available at:
https://github.com/francescomaldonato/RL_VPP_Thesis


## Outline:
This research has the intent to investigate on a sustainable way of life of a general household energy production and storage.
The main goal is to prove a self-sustained energy system with minimum power coming from the grid and expenses. 
The main production consists on PV modules and the storage system is instead based on EVs batteries.
An RL agent will be in charge on managing EVs power resources to guarantee peak shaving and valley filling of the power grid.


## Initialization

- Open the VPP_environment notebook on Google Colaboratory at:
https://colab.research.google.com/github/francescomaldonato/RL_VPP_Thesis/blob/main/VPP_environment.ipynb
- (Recommended for training the agent) Change the runtime type > Hardware accelerator to GPU
- Run the whole notebook

It will automatically clone in the remote machine the repository: https://github.com/francescomaldonato/RL_VPP_Thesis.git

## Authors and acknowledgment
Francesco Maldonato: francesco.maldonato97@gmail.com
