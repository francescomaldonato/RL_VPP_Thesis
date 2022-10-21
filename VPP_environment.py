import os
import numpy as np
import pandas as pd
from random import randrange
import contextlib
from gym import Env
from gym.spaces import Box, Dict, MultiDiscrete

from elvis.simulate import simulate
from elvis.utility.elvis_general import create_time_steps
from elvis.utility.elvis_general import num_time_steps
from dateutil.relativedelta import relativedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def VPP_Scenario_config(yaml_config):
    """
    Function to extrapolate the VPP simulation data from the Elvis YAML config file
    """
    start_date = yaml_config["start_date"]
    end_date = yaml_config["end_date"]
    resolution = yaml_config["resolution"]

    num_households_load = yaml_config["num_households_load"]
    av_max_households_load = yaml_config["av_max_households_load"]
    av_max_energy_price = yaml_config["av_max_energy_price"]
    solar_power = yaml_config["rw_infrastructure"][0]["solar_inverter"]["max_power"]
    wind_power = yaml_config["rw_infrastructure"][1]["wind_inverter"]["max_power"]

    EV_types = yaml_config["vehicle_types"]

    charging_stations_n = len(yaml_config["infrastructure"]["transformers"][0]["charging_stations"])
    EVs_n = yaml_config["num_charging_events"] #per week
    EVs_n_max = int(EVs_n*52.2) #(52.14 weeks in a year)
    EVs_mean_soc = yaml_config["mean_soc"]
    EVs_std_deviation_soc = yaml_config["std_deviation_soc"]
    mean_park = yaml_config["mean_park"]
    std_deviation_park =yaml_config["std_deviation_park"]
    EV_load_max = 0
    EV_load_rated = 0
    for charging_point in yaml_config["infrastructure"]["transformers"][0]["charging_stations"]:
        EV_load_max = EV_load_max + charging_point["max_power"]#kW
        EV_load_rated = EV_load_rated + charging_point["rated_power"]#kW
    EV_load_min = yaml_config["infrastructure"]["transformers"][0]["min_power"]
    houseRWload_max = av_max_households_load

    simulation_param = {
        "start_date": start_date,
        "end_date": end_date,
        "resolution": resolution,

        "num_households": num_households_load,
        "solar_power": solar_power, #kw
        "wind_power": wind_power, #kw
        "EV_types": EV_types,

        "charging_stations_n": charging_stations_n,
        "EVs_n": EVs_n,
        "EVs_n_max": EVs_n_max,
        "mean_park": mean_park, #hours
        "std_deviation_park": std_deviation_park, #std in hours
        "EVs_mean_soc": EVs_mean_soc*100, #% battery on arrival
        "EVs_std_deviation_soc": EVs_std_deviation_soc*100, #Std in kWh
        "EV_load_max": EV_load_max, #kW
        "EV_load_rated": EV_load_rated, #kW
        "EV_load_min": EV_load_min, #kW
        "houseRWload_max": houseRWload_max, #kW
        "av_max_energy_price": av_max_energy_price #€/kWh
    }
    return simulation_param


class VPPEnv(Env):
    """
    VPP environment class based on the openAI gym Env clss
    """
    def __init__(self, VPP_data_input_path, elvis_config_file, simulation_param):
        """
        Initialization function to set all the simulation parameters, variables, action and state spaces.
        """
        #Loading VPP data from path
        VPP_data = pd.read_csv(VPP_data_input_path)
        VPP_data['time'] = pd.to_datetime(VPP_data['time'])
        VPP_data = VPP_data.set_index('time')

        #Costants for all episodes:
        self.elvis_config_file = elvis_config_file
        self.start = simulation_param["start_date"]
        self.end = simulation_param["end_date"]
        self.res = simulation_param["resolution"]

        num_households = simulation_param["num_households"]
        self.solar_power = simulation_param["solar_power"]
        self.wind_power = simulation_param["wind_power"]

        self.EV_types = simulation_param["EV_types"]

        self.charging_stations_n = simulation_param["charging_stations_n"]
        self.EVs_n = simulation_param["EVs_n"]
        self.EVs_n_max = simulation_param["EVs_n_max"]
        self.mean_park = simulation_param["mean_park"] #hours
        self.std_deviation_park = simulation_param["std_deviation_park"] #std in hours
        self.EVs_mean_soc = simulation_param["EVs_mean_soc"]# %translated to kWh
        self.EVs_std_deviation_soc = simulation_param["EVs_std_deviation_soc"]# %translated to kWh
        self.EV_load_max = simulation_param["EV_load_max"]
        #self.EV_load_rated = simulation_param["EV_load_rated"]
        self.charging_point_max_power = self.EV_load_max/self.charging_stations_n #kW
        self.charging_point_rated_power = simulation_param["EV_load_rated"]/self.charging_stations_n #kW
        self.charging_point_min_power = simulation_param["EV_load_min"]
        self.houseRWload_max = simulation_param["houseRWload_max"] + (num_households * 1)
        self.max_energy_price = simulation_param["av_max_energy_price"]
        main_battery_capacity = self.EV_types[0]["battery"]["capacity"]
        self.battery_max_limit = main_battery_capacity - ((main_battery_capacity/100)*0.01) #99.99kWh
        self.battery_min_limit = (main_battery_capacity/100)*0.01 #0.01kWh
        self.DISCHARGE_threshold = 20 #percentage of battery below with the EV can't be discharged
        self.IDLE_DISCHARGE_threshold = 10 #percentage of battery below with the EV can't be discharged, kept idle (must be charged)

        elvis_realisation = elvis_config_file.create_realisation(self.start, self.end, self.res)

        #ELVIS Initial simulation
        #To be updated each episode:
        self.charging_events = elvis_realisation.charging_events
        print(self.charging_events[0], '\n', '...', '\n', self.charging_events[-1], '\n')

        self.current_charging_events = []
        self.simul_charging_events_n = len(self.charging_events)
        self.elvis_time_serie = create_time_steps(elvis_realisation.start_date, elvis_realisation.end_date, elvis_realisation.resolution)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            result = simulate(elvis_realisation)
        
        VPP_data["time"] = self.elvis_time_serie
        VPP_data = VPP_data.set_index("time")
        #Check if environment setting power needs to be rescaled
        if self.solar_power!=16 or self.wind_power!=12 or num_households!=4:
            VPP_data["solar_power"] = VPP_data["solar_power"]/16 * self.solar_power
            VPP_data["wind_power"] = VPP_data["wind_power"]/12 * self.wind_power
            VPP_data["household_power"] = VPP_data["household_power"]/4 * num_households
            VPP_data["House&RW_load"] = VPP_data["household_power"] - VPP_data["solar_power"] - VPP_data["wind_power"]
        VPP_data["RW_power"] = VPP_data["solar_power"] + VPP_data["wind_power"]
        VPP_data["ev_power"] = result.aggregate_load_profile(num_time_steps(elvis_realisation.start_date, elvis_realisation.end_date, elvis_realisation.resolution))
        VPP_data["total_load"] = VPP_data["House&RW_load"] + VPP_data["ev_power"]
        VPP_data["total_cost"] = VPP_data["total_load"] * VPP_data["EUR/kWh"] / 4
        self.prices_serie = VPP_data["EUR/kWh"].values #EUR/kWh
        self.houseRW_load = VPP_data["House&RW_load"].values

        self.VPP_loads = pd.DataFrame({'time':self.elvis_time_serie, "House&RW_load":self.houseRW_load, "household_power":VPP_data["household_power"].values, "solar_power":VPP_data["solar_power"].values, "wind_power":VPP_data["wind_power"].values})
        self.VPP_loads = self.VPP_loads.set_index("time")
        
        self.household_consume = VPP_data["household_power"].sum()/4 #kWh
        self.RW_energy = VPP_data["RW_power"].sum()/4 #kWh
        HRW_array = np.array(self.houseRW_load)
        self.sum_HRW_power = np.sum((HRW_array)/4) #kWh
        self.HRW_overenergy = HRW_array[HRW_array>0].sum()/4 #kWh (Grid energy used)
        self.HRW_underenergy = HRW_array[HRW_array<0].sum()/4 #kWh (RE-to-vehicles unused energy)
        self.self_consumption = self.household_consume - self.HRW_overenergy
        self.selfc_rate = (self.self_consumption / self.RW_energy) * 100
        self.autarky_rate = (self.self_consumption / self.household_consume) * 100
        
        dataset_cost_array = HRW_array * np.array(self.prices_serie)/4 
        self.cost_HRW_power = dataset_cost_array.sum() #€
        self.overcost_HRW_power = dataset_cost_array[dataset_cost_array>0].sum() #€
        self.exp_ev_en_left = self.EVs_mean_soc + (-self.HRW_underenergy/self.simul_charging_events_n)
        #ELVIS
        load_array = np.array(VPP_data["total_load"].values)
        cost_array = np.array(VPP_data["total_cost"].values)
        self.av_Elvis_total_load = np.mean(load_array) #kW
        self.std_Elvis_total_load = np.std(load_array) #kW
        self.sum_Elvis_total_load = load_array.sum()/4 #kWh
        self.Elvis_overconsume = load_array[load_array>0].sum()/4 #kWh
        self.Elvis_underconsume = -load_array[load_array<0].sum()/4 #kWh
        self.Elvis_total_cost = np.sum(cost_array) #€
        self.Elvis_overcost = cost_array[cost_array > 0].sum()
        #Init print out
        print("-DATASET: House&RW_energy_sum=kWh ", round(self.sum_HRW_power,2),
                f", Grid_used_en(grid-import)={round(self.HRW_overenergy,2)}kWh",
                f", autarky-rate={round(self.autarky_rate,1)}",
                f", RE-to-vehicle_unused_en(grid-export)={round(self.HRW_underenergy,2)}kWh",
                f", self-consump.rate={round(self.selfc_rate,1)}",
                ", Total_selling_cost=€ ", round(self.cost_HRW_power,2),
                ", Grid_cost=€ ", round(self.overcost_HRW_power,2))
        print("- ELVIS.Simulation (Av.EV_SOC= ", self.EVs_mean_soc, "%):\n",
            "Sum_Energy=kWh ", round(self.sum_Elvis_total_load,2),
            ", Grid_used_en=kWh ", round(self.Elvis_overconsume,2),
            ", RE-to-vehicle_unused_en=kWh ", round(self.Elvis_underconsume,2),
            ", Total_selling_cost=€ ", round(self.Elvis_total_cost,2),
            ", Grid_cost=€ ", round(self.Elvis_overcost,2),
            ", Charging_events= ", self.simul_charging_events_n,
            "\n- Exp.VPP_goals: Grid_used_en=kWh 0, RE-to-vehicle_unused_en=kWh 0, Grid_cost=€ 0",
            ", Av.EV_en_left=kWh ",round(self.exp_ev_en_left,2))

        #Set VPP session length
        self.tot_simulation_len = len(self.elvis_time_serie)
        self.vpp_length = self.tot_simulation_len
        #empty list init
        self.energy_resources, self.avail_EVs_id, self.ev_power, self.charging_ev_power, self.discharging_ev_power, self.overcost, self.total_cost, self.total_load, self.reward_hist = ([],[],[],[],[],[],[],[],[])
        self.max_total_load = self.EV_load_max + self.houseRWload_max
        self.max_cost = self.max_total_load * self.max_energy_price /4
        #Setting reward functions
        self.set_reward_func()

        self.VPP_data = VPP_data
        self.VPP_actions, self.action_truth_list, self.EVs_energy_at_leaving = ([],[],[])
        #self.lstm_states_list = []
        self.av_EV_energy_left, self.std_EV_energy_left, self.sim_total_load, self.sim_av_total_load, self.sim_std_total_load, self.overconsumed_en, self.underconsumed_en, self.sim_total_cost, self.sim_overcost = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cumulative_reward, self.load_t_reward, self.overconsume_reward, self.underconsume_reward, self.overcost_reward, self.EVs_energy_reward, self.AV_EVs_energy_reward = [0, 0, 0, 0, 0, 0, 0]
        #Initializing state space parameters for the first step [0]
        Init_space_Available_energy_sources = np.zeros(self.charging_stations_n, dtype=np.float32)
        #Init_space_Available_EVs_id = np.zeros(self.charging_stations_n, dtype=np.int32)
        Init_space_ev_power = np.zeros(1,dtype=np.float32)
        Init_space_total_load = np.zeros(1,dtype=np.float32)
        #Init_space_total_cost = np.zeros(1,dtype=np.float32)
        Init_space_total_load[0] = self.houseRW_load[0]
        #Init_space_total_cost[0] = self.houseRW_load[0] * self.prices_serie[0]/4

        self.Init_space = {
            'ev_power': Init_space_ev_power, #EV load range in kW, depends on the infrastructure of the simulation
            'total_load': Init_space_total_load, #Total load range in kW, depends on the household and EV infrastructure of the simulation
            #'total_cost': Init_space_total_cost, #Total cost (EUR) per timestep (15 min) [DELETED]
            'Available_energy_sources': Init_space_Available_energy_sources, #Available energy from Charging stations (EVs connected)
            #'Available_evs_id': Init_space_Available_EVs_id #Available vehicles IDs from charging stations at timestep t [DELETED]
            }
        self.space_0 = self.Init_space
        
        #For plotting battery levels
        self.VPP_energies = Init_space_Available_energy_sources
        # Define constants for Action space options
        self.IDLE = 0
        self.CHARGE = 1
        self.DISCHARGE = 2
        #self.VPP_rated_pwr = 3.7 #kW
        self.possible_actions = 3
        #Action space definition
        self.action_space = MultiDiscrete( self.possible_actions * np.ones(self.charging_stations_n))
        #Actions set definition for action masking
        dims = [self.possible_actions]*self.charging_stations_n
        self.actions_set = np.arange(sum(dims))
        self.invalid_actions_t = np.ones(len(self.actions_set), dtype=bool)
        self.lstm_state = None

        spaces = {
            'ev_power': Box(low=-(self.EV_load_max), high=(self.EV_load_max), shape=(1,), dtype=np.float32), #EV load range in kW, depends on the infrastructure of the simulation
            'total_load': Box(low=-(self.max_total_load) , high= (self.max_total_load), shape=(1,), dtype=np.float32), #Total load range in kW, depends on the household and EV infrastructure of the simulation
            #'total_cost': Box(low=-(self.max_cost), high=(self.max_cost), shape=(1,), dtype=np.float32),#Total cost (EUR) per timestep (15 min) [DELETED]
            'Available_energy_sources': Box(low=0.0, high=100, shape=(self.charging_stations_n,), dtype=np.float32),
            #'Available_evs_id': Box(low=0, high=(np.iinfo(np.int32).max)-1, shape=(self.charging_stations_n,), dtype=np.int32) [DELETED]
            }

        dict_space = Dict(spaces)
        self.observation_space = dict_space
        #Set starting cond.
        self.state = self.Init_space
        self.done = False

    def set_reward_func(self):
        """
        Function to set the reward interpolating functions
        """
        #Step rewards
        #step EV energies -> [0, 50kWh, (75-80kWh), 100kWh]
        self.battery_percentage =[0, self.EVs_mean_soc, 90, 100]
        self.EVs_energy_reward_range = np.array([-300, -40, +150, 50])

        #step Load reward #Normalizing Load range according to datset max load (self.max_total_load = 40 kW)
        load_array = np.array([-80, -30, -15, -4, -1, 0, 0.5, 3.5, 15, 30, 80]) #kW
        self.load_range = load_array
        #self.load_range = (self.max_total_load/100)*np.array([-100, -35, -15, -3, -1.5, 0.1, 3, 15, 35, 100])
        #[-40kW, -12, -6kW, -2kW, -0.4kW, 0, 0.04kW, 2kW, 20kW, 40kW]
        self.load_reward_range = np.array([-50, -30, -15, -5, 0, 15, 0, -5, -20, -40, -80])

        #FINAL REWARDS
        #Av_EV_energy_left reward
        #self.av_energy_left_range = [0, 50, 60, 90, 100]
        self.av_energy_left_range = [0, self.EVs_mean_soc, self.exp_ev_en_left, 100]
        self.av_energy_reward_range = np.array([-50000, -10000, 30000, 10000])

        #Average load #Normalizing av.load according to av_Elvis_load = 6.85 kW
        self.overconsume_range = (self.Elvis_overconsume/100)*np.array([0, 2, 100, 200]) #Elvis over-consume=kWh  43221.9
        #av_load_label = ["0 kWh", "800 kWh", "8000 kWh", "40000 kWh"]
        self.overconsume_reward_range = np.array([1000, 0, -2000, -5000])*20

        #Stand_dev load #Normalizing std.load according to std_Elvis_load = 11.96
        self.underconsume_range = (self.Elvis_underconsume/100)*np.array([0, 20, 100, 200]) #Elvis under-consume=kWh  14842.72
        #std_load_label = ["0 kWh", "2000 kWh","15000 kWh", "30000 kWh"]
        self.underconsume_reward_range = np.array([1000, 0, -3000, -5000])*5

        #total COST #Normalizing total cost according to Elvis_overcost = 2115€
        self.overcost_range = (self.Elvis_overcost/100)*np.array([0, 10, 30, 100, 150]) #Elvis overcost=€  1514.69 
        #cost_label = ["0€", "200€", "800€", "1200€", "2000€"]
        self.overcost_reward_range = np.array([2000, 100, 0, -1000, -2000])*10


    def eval_reward(self, reward, step, new_ev_departures):
        """
        Function to evaluate the agent reward at each timestep of the simulation
        """
        #Load step state variables
        total_load_t = self.total_load[step]
        #energy_resources = self.energy_resources[step] #[DELETED]
    
        #EVs reward: energies available at charging stations
        EVs_energy_reward_t = 0
        #Reward at each timestep for EVs available energy--> confusing for the Agent to learn the policy [DELETED]
        """ for n in range(self.charging_stations_n):
            #1. Check if Evs connected and evaluate reward at station n
            if energy_resources[n] > 0:
                EVs_energy_reward_t += np.interp(energy_resources[n], self.battery_percentage, self.EVs_energy_reward_range)
        self.EVs_energy_reward += EVs_energy_reward_t """

        EVs_energy_leaving_reward_t = 0
        #Apply reward on energy left on vehicle WHEN leaving the station (it accentuate good/bad behaviour)
        for j in range(new_ev_departures):
            energy_left = self.EVs_energy_at_leaving[-1-j]
            EVs_energy_leaving_reward_t += np.interp(energy_left, self.battery_percentage, self.EVs_energy_reward_range)
        self.EVs_energy_reward += EVs_energy_leaving_reward_t
        self.EVs_reward_hist[step-1] = EVs_energy_leaving_reward_t

        #Load reward for each timestep
        load_reward_t = np.interp(total_load_t, self.load_range, self.load_reward_range)
        self.load_t_reward += load_reward_t
        self.load_reward_hist[step-1] = load_reward_t

        reward += (EVs_energy_reward_t + EVs_energy_leaving_reward_t + load_reward_t)
        return reward
    
    def eval_final_reward(self, reward):
        """
        Function to evaluate the final agent reward at the end of the simulation
        """
        #EVs ENERGY reward: Evaluating reward for average energy left in EV leaving
        AV_EVs_energy_reward = np.interp(self.av_EV_energy_left,
                    self.av_energy_left_range,
                    self.av_energy_reward_range)
        self.AV_EVs_energy_reward += AV_EVs_energy_reward

        #LOAD reward:
        #Overconsumed energy reward
        final_overconsume_reward = np.interp(self.overconsumed_en, self.overconsume_range, self.overconsume_reward_range)
        self.overconsume_reward += final_overconsume_reward

        #Underconsumed energy reward
        final_underconsume_reward = np.interp(self.underconsumed_en, self.underconsume_range, self.underconsume_reward_range)
        self.underconsume_reward += final_underconsume_reward

        #OverCOST reward:
        final_overcost_reward = np.interp(self.sim_overcost, self.overcost_range, self.overcost_reward_range)
        self.overcost_reward += final_overcost_reward

        reward += (AV_EVs_energy_reward + final_overconsume_reward + final_underconsume_reward + final_overcost_reward)
        return reward
    
    def action_masks(self):
        """
        Function to evaluate the "invalid" actions the agent should not take at each timestep depending on the step observation
        (used by the Maskable PPO algorithm for training and prediction and by all the algorithms simulation for general behaviour control)
        """
        #self.IDLE = 0
        #self.CHARGE = 1
        #self.DISCHARGE = 2
        step = self.tot_simulation_len - self.vpp_length    
        #loding step variables
        Evs_id_t = self.avail_EVs_id[step]
        EVs_available = 0
        for n in range(self.charging_stations_n):
            if Evs_id_t[n] > 0: EVs_available+=1
        Energy_sources_t = self.energy_resources[step]
        houseRW_load = self.houseRW_load[step+1]
        invalid_actions = []

        for n in range(self.charging_stations_n):
            if Evs_id_t[n] == 0:#if vehicle not present at station n:
                for i in range(1, self.possible_actions): #CHARGE,DISCHARGE invalid
                    invalid_actions.append(n + i*self.charging_stations_n)
            else:
                #IF vehicle present at station n:
                if Energy_sources_t[n] <= self.IDLE_DISCHARGE_threshold:
                    #IDLE,DISCHARGE invalid if battery below 40%
                    for i in range(self.IDLE, self.possible_actions, 2):
                        invalid_actions.append(n + i*self.charging_stations_n)

                elif self.IDLE_DISCHARGE_threshold < Energy_sources_t[n] <= self.DISCHARGE_threshold:
                    #DISCHARGE invalid if battery below 55%
                    invalid_actions.append(n + self.DISCHARGE*self.charging_stations_n)

                elif Energy_sources_t[n] > 91:
                    #CHARGE invalid if battery over 91%
                    invalid_actions.append(n + self.CHARGE*self.charging_stations_n)

                if houseRW_load > 0: #if load positive
                    invalid_actions.append(n + self.CHARGE*self.charging_stations_n) #CHARGE invalid

                    if EVs_available == 1: #if only vehicle available:
                        invalid_actions.append(n) #IDLE invalid
                        if (n + self.DISCHARGE*self.charging_stations_n) in invalid_actions: #if DISCHARGE was invalid
                            invalid_actions.remove(n + self.DISCHARGE*self.charging_stations_n) #DISCHARGE valid
                    elif EVs_available > 1: #if not the only vehicle available:
                        if all( Energy_sources_t[n] >= x for x in [bat_perc for bat_perc in Energy_sources_t if bat_perc > 0]):#if vehicle with most charge
                            invalid_actions.append(n) #IDLE invalid
                            if (n + self.DISCHARGE*self.charging_stations_n) in invalid_actions: #if DISCHARGE is invalid
                                invalid_actions.remove(n + self.DISCHARGE*self.charging_stations_n) #DISCHARGE valid


                elif houseRW_load < 0: #if load is negative
                    invalid_actions.append(n + self.DISCHARGE*self.charging_stations_n) #Discharge invalid
                    #for i in range(self.IDLE, self.possible_actions, 2):
                    #    invalid_actions.append(n + i*self.charging_stations_n) #IDLE,DISCHARGE invalid
                    if EVs_available == 1: #if only vehicle available:
                        invalid_actions.append(n) #IDLE invalid
                        if (n + self.CHARGE*self.charging_stations_n) in invalid_actions: #if CHARGE was invalid
                            invalid_actions.remove(n + self.CHARGE*self.charging_stations_n) #CHARGE valid
                    elif EVs_available > 1: #if not the only vehicle available:
                        if all( Energy_sources_t[n] <= x for x in [bat_perc for bat_perc in Energy_sources_t if bat_perc > 0]):#if vehicle with least charge
                            invalid_actions.append(n) #IDLE invalid
                            if (n + self.CHARGE*self.charging_stations_n) in invalid_actions: #if CHARGE is invalid
                                invalid_actions.remove(n + self.CHARGE*self.charging_stations_n) #CHARGE valid

        invalid_actions = [*set(invalid_actions)]
        self.invalid_actions_t = [action not in invalid_actions for action in self.actions_set]
        return self.invalid_actions_t
    
    def apply_action_on_energy_source(self, step, Energy_sources_t_1, action, total_ev_power_t, ch_station_ideal_pwr):
        """
        Function to apply the agent's chosen action (IDLE,CHARGE,DISCHARGE) to the selected charging station with an EV present.
        the Adaptive power selection calculates the power to get a total zero-load according to the agent's actions chosen.
        """
        #Adaptive power selection
        if ch_station_ideal_pwr > 0:
            if action == self.CHARGE or (Energy_sources_t_1 <= self.IDLE_DISCHARGE_threshold):
                selected_power = ch_station_ideal_pwr
            else: selected_power = self.charging_point_min_power
        elif ch_station_ideal_pwr < 0:
            if (Energy_sources_t_1 <= self.IDLE_DISCHARGE_threshold):
                selected_power = self.charging_point_rated_power
            elif action == self.DISCHARGE:
                selected_power = -ch_station_ideal_pwr
            else: selected_power = self.charging_point_min_power
        else: selected_power = self.charging_point_rated_power
        
        battery_max_limit = self.battery_max_limit #99.9 kWh
        battery_min_limit = self.battery_min_limit #0.1 kWh

        ev_power_t = 0
        #APPLY ACTION on previous energy state:
        if action == self.CHARGE:
            Energy_sources_t = Energy_sources_t_1 + (selected_power * 0.25) #5 kW * 15 min = 1.25 kWh STORING ENERGY
            ev_power_t += selected_power
            if Energy_sources_t > battery_max_limit: #Reached max capacity (kWh)
                ev_power_t -= (Energy_sources_t - battery_max_limit)/0.25
                Energy_sources_t = battery_max_limit
        elif action == self.IDLE:
            if Energy_sources_t_1 <= self.IDLE_DISCHARGE_threshold: #if energy below the Idle-Discharge threshold (10%) --> CHARGE
                Energy_sources_t = Energy_sources_t_1 + (selected_power * 0.25) #5 kW * 15 min = 1.25 kWh STORING ENERGY
                ev_power_t += selected_power
                if Energy_sources_t > battery_max_limit: #Reached max capacity (kWh)
                    ev_power_t -= (Energy_sources_t - battery_max_limit)/0.25
                    Energy_sources_t = battery_max_limit
            #elif Energy_sources_t_1 > self.IDLE_DISCHARGE_threshold: #if energy above the Idle-Discharge threshold (10%) --> IDLE
            else: Energy_sources_t = Energy_sources_t_1 #keep energy constant
        elif action == self.DISCHARGE:
            if Energy_sources_t_1 <= self.IDLE_DISCHARGE_threshold: #if energy below the Idle-Discharge threshold (10%) --> CHARGE
                Energy_sources_t = Energy_sources_t_1 + (selected_power * 0.25) #5 kW * 15 min = 1.25 kWh STORING ENERGY
                ev_power_t += selected_power
                if Energy_sources_t > battery_max_limit: #Reached max capacity (kWh)
                    ev_power_t -= (Energy_sources_t - battery_max_limit)/0.25
                    Energy_sources_t = battery_max_limit
            elif Energy_sources_t_1 > self.DISCHARGE_threshold: #if energy above the Discharge threshold (25%) --> DISCHARGE
                Energy_sources_t = Energy_sources_t_1 - (selected_power * 0.25) #5 kW * 15 min = 1.25 kWh PUSHING ENERGY
                ev_power_t -= selected_power
                if Energy_sources_t < battery_min_limit: #Reached min capacity (kWh)
                    ev_power_t += (battery_min_limit - Energy_sources_t)/0.25
                    Energy_sources_t = battery_min_limit
            #elif Energy_sources_t_1 <= self.DISCHARGE_threshold: #if energy below the Discharge threshold (25%) --> IDLE
            else: Energy_sources_t = Energy_sources_t_1 #keep energy constant
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        #Charge/discharge power series update
        if ev_power_t < 0:
            self.discharging_ev_power[step] += ev_power_t
        elif ev_power_t > 0:
            self.charging_ev_power[step] += ev_power_t
        total_ev_power_t += ev_power_t

        return Energy_sources_t, total_ev_power_t


    def step(self, action):
        """
        Function to update the environment every step along the simulation. Divided in 5 main sections:
        - 0. Actions analysis and Adaptive power evaluation
        - 1. EVs leaving the charging stations check (apply action on those before leaving, then update the present EVs ID array and current_charging_events list)
        - 2. Apply action to each station with an EVs present in the EVs ID array
        - 3. New EVs arrival at the charging stations check (update the present EVs ID array, the simulation charging_events list and the current_charging_events list)
        - 4. Update the VPP environment States and Values
        Final_section: Evaluate final reward, data structures and performance when the simulation is over.
        """
        #Reduce VPP session length by 1 step [Episode init and reset do as step 0, first step = 1]
        self.vpp_length -= 1
        
        #evaluate step
        step = self.tot_simulation_len - self.vpp_length
        time_step = self.elvis_time_serie[step]
    
        #loding step variables
        Evs_id_t = self.avail_EVs_id[step]
        Energy_sources_t = self.energy_resources[step]
        houseRW_load_t = self.houseRW_load[step]
        Evs_id_t_1 = self.avail_EVs_id[step-1] #not used
        Energy_sources_t_1 = self.energy_resources[step-1]

        #Reward Initialization
        reward = 0
        #Variables inititalization for calculations
        total_ev_power_t = 0
        action_truth_array = np.ones(self.charging_stations_n, dtype = bool)
        new_ev_departures = 0
        charging_events_to_remove = []

        #SECTION O. Actions analysis and Adaptive power evaluation (ch_station_ideal_pwr = 0 if no correct actions selected)
        charge_actions, forced_charge_actions, discharge_actions, ch_station_ideal_pwr = (0,0,0,0)
        for n in range(self.charging_stations_n):
            #if Evs_id_t_1[n] > 0: EVs_available+=1
            if action[n] == self.CHARGE: charge_actions+= 1
            elif action[n] == self.DISCHARGE:
                if Energy_sources_t_1[n] > self.DISCHARGE_threshold: discharge_actions+= 1
                if Energy_sources_t_1[n] <= self.IDLE_DISCHARGE_threshold and Evs_id_t_1[n] > 0:
                    charge_actions+= 1
                    forced_charge_actions+= 1
            elif action[n] == self.IDLE:
                if Energy_sources_t_1[n] <= self.IDLE_DISCHARGE_threshold and Evs_id_t_1[n] > 0:
                    charge_actions+= 1
                    forced_charge_actions+= 1
        if houseRW_load_t < 0 and charge_actions > 0:
            ch_station_ideal_pwr = -((houseRW_load_t - discharge_actions*self.charging_point_min_power)/charge_actions)
            if ch_station_ideal_pwr > self.charging_point_max_power: ch_station_ideal_pwr = self.charging_point_max_power
            elif ch_station_ideal_pwr < self.charging_point_min_power: ch_station_ideal_pwr = self.charging_point_min_power
        elif houseRW_load_t > 0 and discharge_actions > 0:
            ch_station_ideal_pwr = -((houseRW_load_t + forced_charge_actions*self.charging_point_rated_power + (charge_actions-forced_charge_actions)*self.charging_point_min_power)/discharge_actions)
            if ch_station_ideal_pwr < -self.charging_point_max_power: ch_station_ideal_pwr = -self.charging_point_max_power
            elif ch_station_ideal_pwr > -self.charging_point_min_power: ch_station_ideal_pwr = -self.charging_point_min_power
        #__END__ SECTION 0

        #SECTION 1. Check if current connected EVs left the charging station
        for charging_event in self.current_charging_events:
            leaving_time_i = charging_event.leaving_time
            if time_step >= leaving_time_i:
                #If vehicle left, set correspondant station ID to zero 
                n = charging_event.station_n
                energy_at_leaving_i, total_ev_power_t = self.apply_action_on_energy_source(step, Energy_sources_t_1[n], action[n], total_ev_power_t, ch_station_ideal_pwr)
                self.EVs_energy_at_leaving.append(energy_at_leaving_i)
                new_ev_departures += 1
                Evs_id_t[n] = int(0)
                charging_events_to_remove.append(charging_event)
            else:
                #If Vehicle still connected, correspondant station ID = EV's ID
                Evs_id_t[charging_event.station_n] = charging_event.id
        for charging_event in charging_events_to_remove:
            self.current_charging_events.remove(charging_event)
            #__END__ SECTION 1

        #SECTION 2. Apply action to each station section 
        for n in range(self.charging_stations_n):
            #1. Check Evs id present and evaluate new Energy available at station n
            if Evs_id_t[n] > 0:
                Energy_sources_t[n], total_ev_power_t = self.apply_action_on_energy_source(step, Energy_sources_t_1[n], action[n], total_ev_power_t, ch_station_ideal_pwr)

            elif Evs_id_t[n] == 0:
                #If no car is connected at station n, available energy = 0
                if Energy_sources_t[n] != 0:
                    raise ValueError("Available_energy_sources table not matching EVs id: state={} where there is an empty station with a certain energy.".format(Energy_sources_t))
            
            #Cheching if invalid actions performed, storing them in a table
            action_code = (self.charging_stations_n*action[n])+n
            action_truth_array[n] = self.invalid_actions_t[action_code]
            #Punishment for invalid actions
            #if action_truth_array[n] == False:
            #    reward += -50

            if Energy_sources_t[n] < 0 or Energy_sources_t[n] > 100:
                #Check if energy sources are out of range
                raise ValueError("Available_energy_sources table out of ranges: state={} which is not part of the state space".format(Energy_sources_t))
            #__END__ SECTION 2

        #Checking free spots
        ch_stations_available = []
        for n in range(self.charging_stations_n):
            if Evs_id_t[n] == 0: ch_stations_available.append(n)
        
        charging_events_to_remove = []
        #SECTION 3. Check if new vehicles arrive at charging stations
        for charging_event in self.charging_events:
            arrival_time_i = charging_event.arrival_time
            #Fixing arrival time at step 0, shifted to step 1
            if step == 1:
                if arrival_time_i == self.elvis_time_serie[0]:
                    arrival_time_i = self.elvis_time_serie[1]
            if arrival_time_i <= time_step:
                if len(ch_stations_available)>0:
                    #If free stations available, pop out free ch_station from list and assign vehicle ID to station
                    n = ch_stations_available.pop(randrange(len(ch_stations_available)))
                    #if type(charging_event.id) != int:
                    charging_event.id = int(charging_event.id[16:])
                    Evs_id_t[n] = charging_event.id
                    vehicle_i = charging_event.vehicle_type.to_dict()
                    soc_i = charging_event.soc
                    battery_i = vehicle_i['battery']
                    #efficiency_i  = battery_i['efficiency'] #Not implemented right now
                    capacity_i  = battery_i['capacity'] #kWh
                    #capacity_i  = 100 #kWh, considering only Tesla Model S
                    energy_i = soc_i * capacity_i #kWh
                    if energy_i < 0.1: #Less than min capacity (kWh) in simulation
                        energy_i = 0.1
                    Energy_sources_t[n] = energy_i
                    charging_event.station_n = n
                    self.current_charging_events.append(charging_event)
                    charging_events_to_remove.append(charging_event)
                    #break
            elif arrival_time_i > time_step:
                break
        for charging_event in charging_events_to_remove:
            self.charging_events.remove(charging_event)
        self.avail_EVs_n[step] = self.charging_stations_n - len(ch_stations_available)
        #__END__ SECTION 3

        #SECTION 4. VPP States and Values updates
        self.ev_power[step] = total_ev_power_t
        self.total_load[step] = houseRW_load_t + total_ev_power_t
        if self.total_load[step] > 0 and self.prices_serie[step] > 0:
            self.overcost[step] = self.total_load[step] * self.prices_serie[step] / 4
        else: self.overcost[step] = 0
        self.total_cost[step] = self.total_load[step] * self.prices_serie[step] / 4
        self.avail_EVs_id[step] = Evs_id_t
        self.energy_resources[step] = Energy_sources_t
        #Evaluate step reward
        reward = self.eval_reward(reward, step, new_ev_departures)
        #VPP Table UPDATE
        self.VPP_actions.append(action)
        self.action_truth_list.append(action_truth_array)
        #self.lstm_states_list.append(self.lstm_state)
        #States UPDATE
        self.state['Available_energy_sources'] = Energy_sources_t
        #self.state['Available_evs_id'] = Evs_id_t #[DELETED]

        ev_power_state = np.zeros(1,dtype=np.float32)
        ev_power_state[0] = total_ev_power_t
        self.state['ev_power'] = ev_power_state
        load_state = np.zeros(1,dtype=np.float32)
        load_state[0] = self.total_load[step]
        self.state['total_load'] = load_state
        #cost_state = np.zeros(1,dtype=np.float32) #[DELETED]
        #cost_state[0] = self.total_cost[step] #[DELETED]
        #self.state['total_cost'] = cost_state #[DELETED]
        #__END__ SECTION 4

        #FINAL_SECTION: Check if VPP is done
        if self.vpp_length <= 1:
            self.done = True
            self.VPP_actions.append(np.zeros(self.charging_stations_n, dtype=np.int32))
            self.action_truth_list.append(np.ones(self.charging_stations_n, dtype = bool))
            #self.lstm_states_list.append(None)
            #Evaluating load sum (overconsumed, underconsumed), std and average up to timestep t for further rewards
            for load in self.total_load:
                self.sim_total_load += load/4 #kWh
                if load >= 0: self.overconsumed_en += load/4 #kWh
                elif load < 0: self.underconsumed_en -= load/4 #kWh
            self.sim_av_total_load = np.mean(self.total_load)
            self.sim_std_total_load = np.std(self.total_load)
            self.sim_overcost = np.sum(self.overcost)
            self.sim_total_cost = np.sum(self.total_cost)
            self.av_EV_energy_left = np.mean(self.EVs_energy_at_leaving)
            self.std_EV_energy_left = np.std(self.EVs_energy_at_leaving)
            charging_events_n = len(self.EVs_energy_at_leaving)
            charging_events_left = len(self.charging_events)
            VPP_loads = self.VPP_loads #Retrieving the VPP loads Dataframe to evaluate autarky and self-consump.
            VPP_loads["charging_ev_power"] = self.charging_ev_power
            VPP_loads["discharging_ev_power"] = self.discharging_ev_power

            #RENEWABLE-SELF-CONSUMPTION evaluation section
            #Households consump. energy not covered from the Renewables
            VPP_loads["house_self-consump."] = VPP_loads["household_power"] - VPP_loads["RE_power"]
            VPP_loads["RE-uncovered_consump."] = VPP_loads["house_self-consump."].mask(VPP_loads["house_self-consump."].lt(0)).fillna(0) #Filter only positive values
            self.house_uncovered_RE = self.VPP_loads["RE-uncovered_consump."].sum()/4 #kWh
            #Energy from the Renewables directly used by the households
            VPP_loads["house_self-consump."] = VPP_loads["household_power"] - VPP_loads["RE-uncovered_consump."]
            self.VPP_house_selfc = VPP_loads["house_self-consump."].sum()/4 #kWh
            #Energy from the Renewables exported to the grid
            VPP_loads["house-unused-RE-power"] = VPP_loads["RE_power"] - VPP_loads["house_self-consump."]
            VPP_loads["self_EV-charging"] = VPP_loads["charging_ev_power"] - VPP_loads["house-unused-RE-power"]
            VPP_loads["RE-grid-export"] = - VPP_loads["self_EV-charging"].mask(VPP_loads["self_EV-charging"].gt(0)).fillna(0) #Filter only negative values
            self.RE_grid_export = VPP_loads["RE-grid-export"].sum()/4 #kWh
            #Energy from the Renewables directly stored in the EVs batteries
            VPP_loads["RE-uncovered_EV-charging"] = VPP_loads["self_EV-charging"].mask(VPP_loads["self_EV-charging"].lt(0)).fillna(0) #Filter only positive values
            VPP_loads["self_EV-charging"] = VPP_loads["charging_ev_power"] - VPP_loads["RE-uncovered_EV-charging"]
            self.VPP_RE2battery = VPP_loads["self_EV-charging"].sum()/4 #kWh

            #EV-DISCHARGING-Power-SELF-CONSUMPTION evaluation section
            #Households consump. grid import (Energy not covered from the EVs discharging power and Renewables)
            VPP_loads["battery-self-consump."] = VPP_loads["RE-uncovered_consump."] - (-VPP_loads["discharging_ev_power"]) #THe discharging EV power is a negative serie
            VPP_loads["house-grid-import"] = VPP_loads["battery-self-consump."].mask(VPP_loads["battery-self-consump."].lt(0)).fillna(0) #Filter only positive values
            self.house_grid_import = VPP_loads["house-grid-import"].sum()/4 #kWh
            #Energy from the EVs discharging power used by the households
            VPP_loads["battery-self-consump."] = VPP_loads["RE-uncovered_consump."] - VPP_loads["house-grid-import"]
            self.VPP_battery_selfc = VPP_loads["battery-self-consump."].sum()/4 #kWh
            #Energy from the EVs discharging power, exported to the grid
            VPP_loads["self_battery-EV-charging"] = (-VPP_loads["discharging_ev_power"]) - VPP_loads["battery-self-consump."] #THe discharging EV power is a negative serie
            VPP_loads["self_battery-EV-charging"] = VPP_loads["RE-uncovered_EV-charging"] - VPP_loads["self_battery-EV-charging"] #ChargingEVs energy not from renwables - (EVs discharging power not used for the house)
            VPP_loads["battery-grid-export"] = - VPP_loads["self_battery-EV-charging"].mask(VPP_loads["self_battery-EV-charging"].gt(0)).fillna(0) #Filter only negative values
            self.battery_grid_export = VPP_loads["battery-grid-export"].sum()/4 #kWh
            #Energy from the grid stored in other EVs batteries
            VPP_loads["grid-import_EV-charging"] = VPP_loads["self_battery-EV-charging"].mask(VPP_loads["self_battery-EV-charging"].lt(0)).fillna(0) #Filter only positive values
            self.EVs_grid_import = VPP_loads["grid-import_EV-charging"].sum()/4 #kWh
            #Energy from the EVs discharging power stored in other EVs batteries
            VPP_loads["self_battery-EV-charging"] = VPP_loads["RE-uncovered_EV-charging"] - VPP_loads["grid-import_EV-charging"]
            self.VPP_EV2battery = VPP_loads["self_battery-EV-charging"].sum()/4 #kWh
            #Rates evaluation
            self.VPP_energy_consumed = self.house_grid_import + self.EVs_grid_import + (self.VPP_house_selfc + self.VPP_battery_selfc + self.VPP_RE2battery + self.VPP_EV2battery)
            self.VPP_autarky_rate = ((self.VPP_house_selfc + self.VPP_battery_selfc + self.VPP_RE2battery + self.VPP_EV2battery) / self.VPP_energy_consumed) * 100
            self.VPP_energy_produced = self.RE_grid_export + self.battery_grid_export + (self.VPP_house_selfc + self.VPP_battery_selfc + self.VPP_RE2battery + self.VPP_EV2battery)
            self.VPP_selfc_rate = ((self.VPP_house_selfc + self.VPP_battery_selfc + self.VPP_RE2battery + self.VPP_EV2battery) / self.VPP_energy_produced) * 100

            #Storing the modified VPP loads Dataframe
            self.VPP_loads = VPP_loads
            #Final reward evaluation
            reward = self.eval_final_reward(reward)
            print("- VPP.Simulation results\n",
                "LOAD_INFO: Sum_Energy=KWh ", round(self.sim_total_load,2),
                f", Grid_used_en(grid-import)={round(self.overconsumed_en,2)}kWh",
                f", Total_demand={round(self.VPP_energy_consumed,2)}kWh",
                f", autarky-rate={round(self.VPP_autarky_rate,1)}",
                f", RE-to-vehicle_unused_en(grid-export)={round(self.underconsumed_en,2)}kWh",
                f", Total_supply={round(self.VPP_energy_produced,2)}kWh",
                f", self-consump.rate={round(self.VPP_selfc_rate,1)}",
                ", Total_selling_cost=€ ", round(self.sim_total_cost,2),
                ", Grid_cost=€ ", round(self.sim_overcost,2),
                "\n",
                "EV_INFO: Av.EV_energy_leaving=kWh ", round(self.av_EV_energy_left,2),
                ", Std.EV_energy_leaving=kWh ", round(self.std_EV_energy_left,2),
                ", EV_departures = ", charging_events_n,
                ", EV_queue_left = ", charging_events_left)

        else:
            self.done = False

        self.reward_hist[step-1] = reward
        #Building final tables
        if self.done == True:
            self.optimized_VPP_data = pd.DataFrame({'time':self.elvis_time_serie, "rewards":self.reward_hist, "ev_power":self.ev_power, "total_load":self.total_load, "total_cost":self.total_cost, "overcost":self.overcost})
            self.optimized_VPP_data = self.optimized_VPP_data.set_index("time")
            
            self.action_truth_table = np.stack(self.action_truth_list)
            self.Evs_id_table = np.stack(self.avail_EVs_id)
            self.VPP_energies = np.stack(self.energy_resources)
            self.VPP_table = pd.DataFrame(self.VPP_energies)
            self.VPP_table["time"] = self.elvis_time_serie
            self.VPP_table = self.VPP_table.set_index("time")
            self.VPP_table["EVs_id"] = self.avail_EVs_id
            self.VPP_table["actions"] = self.VPP_actions
            self.VPP_table["mask_truth"] = self.action_truth_list
            self.VPP_table["ev_charged_pwr"] = self.charging_ev_power
            self.VPP_table["ev_discharged_pwr"] = self.discharging_ev_power
            self.VPP_table["load"] = self.total_load
            self.VPP_table["load_reward"] = self.load_reward_hist
            self.VPP_table["EV_reward"] = self.EVs_reward_hist
            self.VPP_table["rewards"] = self.reward_hist
            #self.VPP_table["states"] = self.lstm_states_list
            self.cumulative_reward = np.sum(self.reward_hist)
            self.load_t_reward = np.sum(self.load_reward_hist)
            self.EVs_energy_reward = np.sum(self.EVs_reward_hist)
            self.quick_results = np.array([str(self.EVs_n)+"_EVs", self.underconsumed_en, self.overconsumed_en, self.sim_overcost, self.av_EV_energy_left, self.cumulative_reward])
            print(f"SCORE:  Cumulative_reward= {round(self.cumulative_reward,2)} - Step_rewars (load_t= {round(self.load_t_reward,2)}, EVs_energy_t= {round(self.EVs_energy_reward,2)})\n",
                    f"- Final_rewards (Av.EVs_energy= {round(self.AV_EVs_energy_reward,2)}, Grid_used_en= {round(self.overconsume_reward,2)}, RE-to-vehicle_unused_en= {round(self.underconsume_reward,2)}, Grid_cost= {round(self.overcost_reward,2)})")
            #__END__ FINAL SECTION
        #set placeholder for info
        info = {}
        #return step information
        return self.state, reward, self.done, info

    def render(self, mode = 'human'):
        """
        Rendering function not implemented.
        """
        #implement visualization
        pass

    def reset(self):
        """
        Reset Environment function to be ready for new simulation. Divided in 3 main sections:
        - 1. Create new ELVIS simulation for EVs charging events
        - 2. Reset VPP simulation dataset series applying noise on the excrated original dataset instances (not overwriting)
        - 3. Reset VPP simulation tables and lists to zero or empty to be filled
        """
        #SECTION 1. Create new ELVIS simulation
        elvis_config_file = self.elvis_config_file
        elvis_realisation = elvis_config_file.create_realisation(self.start, self.end, self.res)
        self.charging_events = elvis_realisation.charging_events
        if self.current_charging_events != []:
            current_charging_events = self.current_charging_events
            for i in range(len(current_charging_events)):
                current_charging_events[i].leaving_time = current_charging_events[i].leaving_time - relativedelta(years=1)
            self.current_charging_events = current_charging_events
        self.simul_charging_events_n = len(self.charging_events)
        #Evaluate av.EV energy left with Elvis
        Elvis_av_EV_energy_left, n_av = [0, 0]
        for charging_event in self.charging_events:
            n_av += 1
            vehicle_i = charging_event.vehicle_type.to_dict()
            soc_i = charging_event.soc
            battery_i = vehicle_i['battery']
            #efficiency_i  = battery_i['efficiency'] #Not implemented right now
            capacity_i  = battery_i['capacity'] #kWh
            #capacity_i  = 100 #kWh, considering only Tesla Model S
            energy_i = soc_i * capacity_i #kWh
            charging_time = charging_event.leaving_time - charging_event.arrival_time
            final_energy = energy_i + ((charging_time.total_seconds()/3600) * self.charging_point_max_power)
            if final_energy > capacity_i: final_energy = capacity_i #kWh
            Elvis_av_EV_energy_left = (final_energy + (n_av-1)*Elvis_av_EV_energy_left)/n_av 
        self.Elvis_av_EV_energy_left = Elvis_av_EV_energy_left
        VPP_data = self.VPP_data
        #self.prices_serie = VPP_data["EUR/kWh"].values #EUR/kWh #[DELETED]
        #__END__ SECTION 1
        
        #SECTION 2. Reset VPP simulation data applying noise on the original dataset
        #Data remaining constant: VPP_data["household_power"], VPP_data["solar_power"], VPP_data["wind_power"], VPP_data["EUR/kWh"]:
        mu, sigma = 0, (self.max_energy_price/100)*7 #Mean, standard deviation (self.max_energy_price= 0.13 €/kWh --> 7% = 0.0091)
        price_noise = np.random.normal(mu, sigma, self.tot_simulation_len) # creating a noise with the same dimension of the dataset length
        mu, sigma = 0, (self.houseRWload_max/100)*4 #Mean, standard deviation (self.houseRWload_max= 10kW --> 2% = 0.4 kW)
        load_noise = np.random.normal(mu, sigma, self.tot_simulation_len) # creating a noise with the same dimension of the dataset length
        self.VPP_loads["solar_power"] = VPP_data["solar_power"] - load_noise/3
        self.VPP_loads["wind_power"] = VPP_data["wind_power"] - load_noise/3
        self.VPP_loads["RE_power"] = self.VPP_loads["solar_power"] + self.VPP_loads["wind_power"]
        self.VPP_loads["household_power"] = VPP_data["household_power"] + load_noise/3

        VPP_data["House&RW_load"] = (VPP_data["household_power"] - VPP_data["solar_power"] - VPP_data["wind_power"]) + load_noise
        #Updating series values from noisy table
        self.prices_serie = list(VPP_data["EUR/kWh"].values + price_noise) #EUR/kWh
        self.houseRW_load = VPP_data["House&RW_load"].values  

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            result = simulate(elvis_realisation)
        VPP_data["ev_power"] = result.aggregate_load_profile(num_time_steps(elvis_realisation.start_date, elvis_realisation.end_date, elvis_realisation.resolution))
        #VPP_data["ev_power"].plot()
        VPP_data["total_load"] = VPP_data["House&RW_load"] + VPP_data["ev_power"]
        VPP_data["total_cost"] = VPP_data["total_load"] * VPP_data["EUR/kWh"]/4
        load_array = np.array(VPP_data["total_load"].values)
        cost_array = np.array(VPP_data["total_cost"].values)
        VPP_data["overcost"] = VPP_data["total_cost"]
        VPP_data["overcost"].mask( VPP_data["overcost"] < 0, 0 , inplace=True)
        #Elvis RE2house
        VPP_data["Elvis_RE2house"] = VPP_data["House&RW_load"].mask(VPP_data["House&RW_load"].lt(0)).fillna(0) #Filter only positive values
        VPP_data["Elvis_RE2house"] = self.VPP_loads["household_power"] - VPP_data["Elvis_RE2house"]
        self.Elvis_RE2house_en = VPP_data["Elvis_RE2house"].sum()/4 #kWh
        #Elvis Grid2EV
        VPP_data["Elvis_RE2EV"] = - VPP_data["House&RW_load"].mask(VPP_data["House&RW_load"].gt(0)).fillna(0) #Filter only negative values
        VPP_data["Elvis_RE2EV"] = VPP_data["ev_power"] - VPP_data["Elvis_RE2EV"]
        VPP_data["Elvis_Grid2EV"] = VPP_data["Elvis_RE2EV"].mask(VPP_data["Elvis_RE2EV"].lt(0)).fillna(0) #Filter only positive values
        self.Elvis_Grid2EV_en = VPP_data["Elvis_Grid2EV"].sum()/4 #kWh
        #Elvis RE2EV
        VPP_data["Elvis_RE2EV"] = VPP_data["ev_power"] - VPP_data["Elvis_Grid2EV"]
        self.Elvis_RE2EV_en = VPP_data["Elvis_RE2EV"].sum()/4 #kWh

        self.av_Elvis_total_load = np.mean(load_array) #kW
        self.std_Elvis_total_load = np.std(load_array) #kW
        self.sum_Elvis_total_load = load_array.sum()/4 #kWh
        self.Elvis_overconsume = load_array[load_array>0].sum()/4 #kWh
        self.Elvis_underconsume = -load_array[load_array<0].sum()/4 #kWh
        self.Elvis_total_cost = cost_array.sum() #€
        self.Elvis_overcost = cost_array[cost_array > 0].sum()
        #Elvis self-consumption and autarky eval
        self.Elvis_en_produced = self.Elvis_underconsume + (self.Elvis_RE2house_en + self.Elvis_RE2EV_en)
        self.Elvis_selfc_rate = (self.Elvis_RE2house_en + self.Elvis_RE2EV_en) / self.Elvis_en_produced
        self.Elvis_en_consumed = self.Elvis_overconsume + (self.Elvis_RE2house_en + self.Elvis_RE2EV_en)
        self.Elvis_autarky_rate = (self.Elvis_RE2house_en + self.Elvis_RE2EV_en) / self.Elvis_en_consumed
        #Reset environment printout:
        print("- ELVIS.Simulation (Av.EV_SOC= ", self.EVs_mean_soc, "%):\n",
            "Sum_Energy=kWh ", round(self.sum_Elvis_total_load,2),
            f", Grid_used_en(grid-import)={round(self.Elvis_overconsume,2)}kWh",
            f", Total_demand={round(self.Elvis_en_consumed,2)}kWh",
            f", autarky-rate={round(self.Elvis_autarky_rate,1)}",
            f", RE-to-vehicle_unused_en(grid-export)={round(self.Elvis_underconsume,2)}kWh",
            f", Total_supply={round(self.Elvis_en_produced,2)}kWh",
            f", self-consump.rate={round(self.Elvis_selfc_rate,1)}",
            ", Grid_cost=€ ", round(self.Elvis_overcost,2),
            ", Total_selling_cost=€ ", round(self.Elvis_total_cost,2),
            ", Av.EV_en_left=kWh ", round(Elvis_av_EV_energy_left,2),
            ", Charging_events= ", self.simul_charging_events_n,
            "\n- VPP_goal_upper_limit: Grid_used_en=kWh 0, RE-to-vehicle_unused_en=kWh 0, Grid_cost=€ 0",
            ", Av.EV_en_left=kWh ",round(self.exp_ev_en_left,2))
        #__END__ SECTION 2
        
        #SECTION 3. Reset VPP simulation tables and lists to be filled
        #Setting reward functions
        self.set_reward_func()
        #Reset VPP session length
        self.vpp_length = self.tot_simulation_len
        self.energy_resources, self.avail_EVs_id, self.avail_EVs_n, self.ev_power, self.charging_ev_power, self.discharging_ev_power , self.total_cost, self.overcost, self.total_load, self.reward_hist, self.EVs_reward_hist, self.load_reward_hist = ([],[],[],[],[],[],[],[],[],[],[],[])
        #build EV series (Avail_en. and IDs)
        for i in range(len(self.elvis_time_serie)):
            self.energy_resources.append(np.zeros(self.charging_stations_n, dtype=np.float32))
            self.avail_EVs_id.append(np.zeros(self.charging_stations_n, dtype=np.int32))
            self.avail_EVs_n.append(0)
            self.ev_power.append(0.0)
            self.charging_ev_power.append(0.0)
            self.discharging_ev_power.append(0.0)
            self.total_cost.append(0.0)
            self.overcost.append(0.0)
            self.total_load.append(0.0)
            self.reward_hist.append(0)
            self.EVs_reward_hist.append(0)
            self.load_reward_hist.append(0)
        
        self.total_load[0] = self.houseRW_load[0]
        self.total_cost[0] = self.total_load[0] * self.prices_serie[0]/4
        self.energy_resources[0] = self.Init_space["Available_energy_sources"]
        #self.avail_EVs_id[0] = self.Init_space['Available_evs_id'] #[DELETED]
        self.VPP_data = VPP_data
        self.optimized_VPP_data = pd.DataFrame({'time':self.elvis_time_serie, "rewards":self.reward_hist, "ev_power":self.ev_power, "total_load":self.total_load, "total_cost":self.total_cost})
        self.optimized_VPP_data = self.optimized_VPP_data.set_index("time")
        #self.lstm_states_list = []
        self.VPP_actions, self.action_truth_list, self.EVs_energy_at_leaving= ([],[],[]) 
        self.av_EV_energy_left, self.std_EV_energy_left, self.sim_total_load, self.sim_av_total_load, self.sim_std_total_load, self.overconsumed_en, self.underconsumed_en, self.sim_total_cost, self.sim_overcost = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cumulative_reward, self.load_t_reward, self.overconsume_reward, self.underconsume_reward, self.overcost_reward, self.EVs_energy_reward, self.AV_EVs_energy_reward = [0, 0, 0, 0, 0, 0, 0]
        #For plotting battery levels
        #self.VPP_energies = self.Init_space["Available_energy_sources"] #[DELETED]
        self.invalid_actions_t = np.ones(len(self.actions_set), dtype=bool)
        self.VPP_table = []
        self.quick_results = []
        #Set starting cond.
        self.state = self.Init_space
        #reset vpp session time
        self.vpp_length = self.tot_simulation_len
        self.done = False
        print("Simulating VPP....")
        #__END__ SECTION 3
        return self.state
    
    def save_VPP_table(self, save_path='data/environment_optimized_output/VPP_table.csv'):
        """
        Method to save the VPP optimized simulation data.
        """
        self.VPP_table.to_csv(save_path)
        return self.VPP_table
    
    def plot_ELVIS_data(self):
        """
        Method to plot and visualize the ELVIS simulation input data for the EVs infrastructure.
        """
        #Weekly arrival distribution simulation
        weekly_distribution = self.elvis_config_file.arrival_distribution
        time_frame = self.elvis_time_serie[0:len(weekly_distribution)*4:4]

        EV_battery_capacities,models = ([], [])
        for EV_type in self.EV_types:
            EV_battery_capacities.append(EV_type["battery"]["capacity"])
            #brand.append()
            models.append(str(EV_type['brand'])+str(EV_type['model']))
        
        Elvis_data_fig = make_subplots(subplot_titles=('EVs arrival distribution (weekly)','Simulation parameters', 'EV models', 'Rated powers'),
                            rows=2, cols=2,
                            specs=[[{"secondary_y": False},{"type": "table"}],
                                    [{"secondary_y": False},{"secondary_y": False}]])
        
        Elvis_data_fig.add_trace(
            go.Scatter(x=time_frame, y=weekly_distribution, name="EVs_arrival distribution"),
            row=1, col=1, secondary_y=False)
        
        table_data = [['EV_arrivals(W)','mean_park(h)','mean_park+std','mean_park-std'],[self.EVs_n, self.mean_park, self.mean_park+self.std_deviation_park, self.mean_park-self.std_deviation_park]]
        Elvis_data_fig.add_trace(go.Table(
                                    columnorder = [1,2],
                                    columnwidth = [80,400],
                                    header = dict(
                                        values = [['Parameters'],
                                                    ['Values']],
                                        fill_color='#04cc98',
                                        align=['left','center'],
                                    ),
                                    cells=dict(
                                        values=table_data,
                                        fill=dict(color=['royalblue', 'white']),
                                        align=['left', 'center'],
                                        #height=30
                                    )), row=1, col=2)
                                        
        Elvis_data_fig.add_trace(go.Bar(x=[models[0],'arrival Av.soc','Av.soc-std','Av.soc+std'], y=[EV_battery_capacities[0], self.EVs_mean_soc, (self.EVs_mean_soc-self.EVs_std_deviation_soc), (self.EVs_mean_soc+self.EVs_std_deviation_soc)], marker_color = ['#d62728','#bcbd22','#7f7f7f','#7f7f7f']),
                            row=2, col=1)
        
        rated_powers_x = ['solar max', 'wind max', 'EVs load max', 'ch.point max', 'houseRWload max']
        rated_powers_y = [self.solar_power, self.wind_power, self.EV_load_max, self.charging_point_max_power, self.houseRWload_max]
        marker_color = ['#95bf00', '#1ac6ff', '#ee5940', '#7f7f7f', 'orange']
        
        Elvis_data_fig.add_trace(go.Bar(x=rated_powers_x, y=rated_powers_y, marker_color=marker_color),
                            row=2, col=2)
        
        Elvis_data_fig['layout']['yaxis1'].update(title='Probability')
        Elvis_data_fig['layout']['yaxis2'].update(title='Battery capacity (kWh)')
        Elvis_data_fig['layout']['yaxis3'].update(title='kW')
        #Elvis_data_fig['layout']['legend'].update(title=f'Cumulat.Reward= {round(self.cumulative_reward,2)}')
        Elvis_data_fig.update_layout(title_text='ELVIS simulation input data', width=1500,height=550, showlegend = False)
        return Elvis_data_fig
    
    def plot_VPP_input_data(self):
        """
        Method to plot and visualize the VPP environment input dataset.
        """
        #Optimized VPP simulation graphs
        VPP_data_fig = make_subplots(
                            subplot_titles=('Households and Renewables power over time','Households+RW sources Load over time','Energy cost over time'),
                            rows=3, cols=1, shared_xaxes=True,
                            specs=[[{"secondary_y": False}],
                                    [{"secondary_y": False}],
                                    [{"secondary_y": False}]])

        # Top graph
        VPP_data_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["household_power"], name="household_power",line={'color':'#5c5cd6'}, stackgroup='consumed'),
            row=1, col=1, secondary_y=False)
        VPP_data_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_data["solar_power"], name="solar_power",line={'color':'#95bf00'}, stackgroup='produced'),
            row=1, col=1, secondary_y=False)
        VPP_data_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_data["wind_power"], name="wind_power",line={'color':'#1ac6ff'}, stackgroup='produced'),
            row=1, col=1, secondary_y=False)
        
        # Center graph
        VPP_data_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["House&RW_load"], name="House&RW_load",line={'color': 'orange'}, stackgroup='summed'),
            row=2, col=1, secondary_y=False)

        #Down graph
        VPP_data_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["EUR/kWh"], name="EUR/kWh",line={'color':'rgb(210, 80, 75)'}, stackgroup='cost'),
            row=3, col=1, secondary_y=False)

        VPP_data_fig['layout']['yaxis1'].update(title='kW')
        VPP_data_fig['layout']['yaxis2'].update(title='kW')
        VPP_data_fig['layout']['yaxis3'].update(title='€/kWh')
        VPP_data_fig['layout']['legend'].update(title='Time series')
        VPP_data_fig.update_layout(title_text='VPP simulation input data', width=1500,height=700, xaxis3_rangeslider_visible=True, xaxis3_rangeslider_thickness=0.05, xaxis_range=["2022-06-01 00:00:00", "2022-07-01 00:00:00"])
        VPP_data_fig.update_xaxes(range=["2022-06-01 00:00:00", "2022-06-11 00:00:00"], row=1, col=1)
        VPP_data_fig.update_xaxes(range=["2022-06-01 00:00:00", "2022-06-11 00:00:00"], row=2, col=1)
        VPP_data_fig.update_xaxes(range=["2022-06-01 00:00:00", "2022-06-11 00:00:00"], row=3, col=1)        
        #VPP_data_fig.show()
        return VPP_data_fig
    
    def plot_reward_functions(self):
        """
        Method to plot and visualize the RL agent reward functions.
        """
        #Step rewards
        battery_x = np.linspace(0, 100, 200)
        battery_y = np.interp(battery_x, self.battery_percentage, self.EVs_energy_reward_range)

        load_x = np.linspace(self.load_range[0], self.load_range[-1], 10000)
        load_y = np.interp(load_x, self.load_range, self.load_reward_range)

        #Final rewards
        final_battery_y = np.interp(battery_x, self.av_energy_left_range, self.av_energy_reward_range)

        overconsume_x = np.linspace(self.overconsume_range[0], self.overconsume_range[-1], 200)
        overconsume_y = np.interp(overconsume_x, self.overconsume_range, self.overconsume_reward_range)

        underconsume_x = np.linspace(self.underconsume_range[0], self.underconsume_range[-1], 200)
        underconsume_y = np.interp(underconsume_x, self.underconsume_range, self.underconsume_reward_range)

        cost_x = np.linspace(self.overcost_range[0], self.overcost_range[-1], 200)
        cost_y = np.interp(cost_x, self.overcost_range, self.overcost_reward_range)

        rewards_fig = make_subplots(subplot_titles=('Step EVs energy (when leaving) reward f.','Step load reward f.', 'Final Grid energy used reward f.', 'Final Av.EVs-departure energy reward f.', 'Final Overcost reward f.', 'Final RE-to-vehicle unused energy reward f.'),
                            rows=2, cols=3,
                            specs=[[{"secondary_y": False},{"secondary_y": False},{"secondary_y": False}],
                                    [{"secondary_y": False},{"secondary_y": False},{"secondary_y": False}]])
        
        rewards_fig.add_trace(go.Scatter(x=battery_x, y=battery_y, name="step_ev_energy", stackgroup='1'),
                            row=1, col=1, secondary_y=False)
        rewards_fig.add_trace(go.Scatter(x=load_x, y=load_y, name="step_load", stackgroup='1'),
                            row=1, col=2, secondary_y=False)
        rewards_fig.add_trace(go.Scatter(x=overconsume_x, y=overconsume_y, name="final_Grid_used_en", stackgroup='1'),
                            row=1, col=3, secondary_y=False)
        rewards_fig.add_trace(go.Scatter(x=battery_x, y=final_battery_y, name="final_Av.ev_energy", stackgroup='1'),
                            row=2, col=1, secondary_y=False)
        rewards_fig.add_trace(go.Scatter(x=cost_x, y=cost_y, name="final_Grid-cost", stackgroup='1'),
                            row=2, col=2, secondary_y=False)
        rewards_fig.add_trace(go.Scatter(x=underconsume_x, y=underconsume_y, name="final_RE-to-vehicle_unused_en", stackgroup='1'),
                            row=2, col=3, secondary_y=False)
        
        
        rewards_fig['layout']['xaxis1'].update(title='Battery% (kWh)')
        rewards_fig['layout']['xaxis2'].update(title='kW')
        rewards_fig['layout']['xaxis3'].update(title='kWh')
        rewards_fig['layout']['xaxis4'].update(title='Battery% (kWh)')
        rewards_fig['layout']['xaxis5'].update(title='€')
        rewards_fig['layout']['xaxis6'].update(title='kWh')
        rewards_fig['layout']['yaxis1'].update(title='Step reward')
        rewards_fig['layout']['yaxis2'].update(title='Step reward')
        rewards_fig['layout']['yaxis3'].update(title='Final reward')
        rewards_fig['layout']['yaxis4'].update(title='Final reward')
        rewards_fig['layout']['yaxis5'].update(title='Final reward')
        rewards_fig['layout']['yaxis6'].update(title='Final reward')
        rewards_fig.update_layout(title_text='Reward functions', width=1500,height=700, showlegend = False)
        #rewards_fig.show()
        return rewards_fig
        
    def plot_Dataset_autarky(self):
        """
        Method to plot and visualize the autarky and self-consumption
        in the plain dataset.
        """
        en_demand = self.household_consume
        en_supply = self.RW_energy
        
        selfc_rate = (self.self_consumption / en_supply) * 100
        autarky_rate = (self.self_consumption / en_demand) * 100
        selfc_labels = ["RE2grid-export", "RE2house-self"]
        selfc_values = [-self.HRW_underenergy, self.self_consumption]
        autarky_labels = ["Grid2house-import", "RE2house-self"]
        autarky_values = [self.HRW_overenergy, self.self_consumption]
        # Create subplots: use 'domain' type for Pie subplot
        fig = make_subplots(subplot_titles=(f'Self-consumption rate: {round(selfc_rate,1)}%', f'Autarky rate: {round(autarky_rate,1)}%'),
                            rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        
        fig.add_trace(go.Pie(labels=selfc_labels, values=selfc_values, name="self-consumption", textinfo='label+value+percent', pull=[0.1, 0]),
                      1, 1)
        fig.add_trace(go.Pie(labels=autarky_labels, values=autarky_values, name="autarky", textinfo='label+value+percent', pull=[0.1, 0]),
                      1, 2)
        
        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.2, hoverinfo="label+value+percent")
        
        fig.update_layout(
            title_text="Data-set Autarky and self-consumption. \nEnergy measuring unit: kWh"+f'\nSupply-energy:{round(en_supply,1)}kWh.     '+ f'\nDemand-energy:{round(en_demand,1)}kWh.',
            # Add annotations in the center of the donut pies.
            #annotations=[dict(text=f'Supply-energy:{round(en_supply,1)}kWh', x=0.18, y=0.5, font_size=20, showarrow=False),
            #             dict(text=f'Demand-energy:{round(en_demand,1)}kWh', x=0.82, y=0.5, font_size=20, showarrow=False)],
            width=1500,height=550,
            showlegend = False)

        #fig.show()
        return fig
        

    def plot_VPP_autarky(self):
        """
        Method to plot and compare the autarky and self-consumption in the Elvis uncontrolled-charging
        simulation and in the VPP simulation with controlled-charging actions.
        """
        Elvis_selfc_labels = ["RE2grid-export", "RE2house-self", "RE2EVs-self"]
        Elvis_selfc_values = [self.Elvis_underconsume, self.Elvis_RE2house_en, self.Elvis_RE2EV_en]
        Elvis_autarky_labels = ["Grid2house-import", "Grid2EV-import", "RE2house-self","RE2EVs-self"]
        Elvis_autarky_values = [(self.Elvis_overconsume-self.Elvis_Grid2EV_en), self.Elvis_Grid2EV_en, self.Elvis_RE2house_en, self.Elvis_RE2EV_en]
        
        VPP_selfc_labels = ["RE2grid-export", "EV2grid-export", "RE2house-self", "RE2EVs-self", "EV2house-self", "EV2EV-transf."]
        VPP_selfc_values = [self.RE_grid_export, self.battery_grid_export, self.VPP_house_selfc, self.VPP_RE2battery, self.VPP_battery_selfc, self.VPP_EV2battery]
        VPP_autarky_labels = ["Grid2house-import", "Grid2EV-import", "RE2house-self", "RE2EVs-self", "EV2house-self", "EV2EV-transf."]
        VPP_autarky_values = [self.house_grid_import, self.EVs_grid_import, self.VPP_house_selfc, self.VPP_RE2battery, self.VPP_battery_selfc, self.VPP_EV2battery]

        # Create subplots
        fig = make_subplots(subplot_titles=('Elvis simulation', 'Elvis simulation',
                                            'VPP simulation', 'VPP simulation'),
                            rows=2, cols=2,
                            specs=[[{'type':'domain'}, {'type':'domain'}],
                                    [{'type':'domain'}, {'type':'domain'}]])
        
        fig.add_trace(go.Pie(labels=Elvis_selfc_labels, values=Elvis_selfc_values, name="elvis_self-consumption", textinfo='label+value+percent', pull=[0.1, 0, 0]),
                      1, 1)
        fig.add_trace(go.Pie(labels=Elvis_autarky_labels, values=Elvis_autarky_values, name="elvis_autarky", textinfo='label+value+percent', pull=[0.1, 0.1, 0, 0]),
                      1, 2)

        fig.add_trace(go.Pie(labels=VPP_selfc_labels, values=VPP_selfc_values, name="VPP_self-consumption", textinfo='label+value+percent', pull=[0.1, 0.1, 0, 0, 0]),
                      2, 1)
        fig.add_trace(go.Pie(labels=VPP_autarky_labels, values=VPP_autarky_values, name="VPP_autarky", textinfo='label+value+percent', pull=[0.1, 0.1, 0, 0, 0, 0]),
                      2, 2)
        
        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.2, hoverinfo="label+value+percent")
        
        fig.update_layout(
            #title_text="Data-set Autarky and self-consumption",
            # Add annotations in the center of the donut pies.
            annotations=[#dict(text='Elvis simulation', x=0.25, y=0.95, font_size=14, showarrow=False),
                         dict(text=f'Self-consumption rate: {round(self.Elvis_selfc_rate,1)}%', x=0.4, y=0.88, font_size=12, showarrow=False),
                         dict(text=f'Supply-en: {round(self.Elvis_en_produced,1)} kWh', x=0.4, y=0.83, font_size=12, showarrow=False),
                         #dict(text='Elvis simulation', x=0.65, y=0.95, font_size=14, showarrow=False),
                         dict(text= f'Autarky rate: {round(self.Elvis_autarky_rate,1)}%', x=1, y=0.88, font_size=12, showarrow=False),
                         dict(text= f'Demand-en:{round(self.Elvis_en_consumed,1)}kWh', x=1, y=0.83, font_size=12, showarrow=False),
                         #dict(text='VPP simulation', x=0.05, y=0.35, font_size=14, showarrow=False),
                         dict(text=f'Self-consumption rate: {round(self.VPP_selfc_rate,1)}%', x=0.4, y=0.38, font_size=12, showarrow=False),
                         dict(text=f'Supply-en: {round(self.VPP_energy_produced,1)} kWh', x=0.4, y=0.33, font_size=12, showarrow=False),
                         #dict(text='VPP simulation', x=0.65, y=0.35, font_size=14, showarrow=False),
                         dict(text= f'Autarky rate: {round(self.VPP_autarky_rate,1)}%', x=1, y=0.38, font_size=12, showarrow=False),
                         dict(text= f'Demand-en:{round(self.VPP_energy_consumed,1)}kWh', x=1, y=0.33, font_size=12, showarrow=False)],

            width=1550,height=800,
            showlegend = False)

        #fig.show()
        return fig
    
    def plot_VPP_energies(self):
        """
        Method to plot and visualize the available energy levels present at the charging stations
        during the VPP simulation with controlled charging actions.
        """
        #Plot energy available in the charging points over time
        self.VPP_energies = pd.DataFrame(self.VPP_energies)
        self.VPP_energies["time"] = self.elvis_time_serie
        self.VPP_energies = self.VPP_energies.set_index("time")
        VPP_energies_fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": False}]])
        for n in range(self.charging_stations_n):
            station = str(n)
            VPP_energies_fig.add_trace(go.Scatter(x=self.elvis_time_serie, y=self.VPP_energies[n], name=f"charging station {station}", stackgroup=f"{station}"),
                                    row=1, col=1, secondary_y=False)

        VPP_energies_fig.add_trace(go.Scatter(x=self.elvis_time_serie, y=[self.av_EV_energy_left-self.std_EV_energy_left]*self.tot_simulation_len, line={'color':'lightgrey'},
        name="-Std_EV_energy_left"), row=1, col=1, secondary_y=False)

        VPP_energies_fig.add_trace(go.Scatter(x=self.elvis_time_serie, y=[self.av_EV_energy_left+self.std_EV_energy_left]*self.tot_simulation_len, line={'color':'lightgrey'},
        name="+Std_EV_energy_left"), row=1, col=1, secondary_y=False)

        VPP_energies_fig.add_trace(go.Scatter(x=self.elvis_time_serie, y=[self.av_EV_energy_left]*self.tot_simulation_len, line={'color':'#bcbd22'},
        name="Av_EV_energy_left"), row=1, col=1, secondary_y=False)
            
        VPP_energies_fig['layout']['yaxis1'].update(title='kWh')
        VPP_energies_fig.update_layout(title_text='VPP available energies at EV charging points', width=1500,height= 550, xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.05, xaxis_range=["2022-06-01 00:00:00", "2022-07-01 00:00:00"])
        #VPP_energies_fig.show()
        return VPP_energies_fig

    def plot_Elvis_results(self):
        """
        Method to plot and visualize the ELVIS simulation results (load, EV power and overcost) with uncontrolled charging.
        """
        #Elvis simulation graphs
        Elvis_fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

        # Top graph
        #Elvis_fig.add_trace(
        #    go.Scatter(x=self.elvis_time_serie, y=[0]*self.tot_simulation_len,line={'color':'#00174f'}, name="zero_load"),
        #    row=1, col=1, secondary_y=False)
        
        """ Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_data["solar_power"], name="solar_power",line={'color':'#95bf00'}),
            row=1, col=1, secondary_y=False)
        Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_data["wind_power"], name="wind_power",line={'color':'#1ac6ff'}),
            row=1, col=1, secondary_y=False)
        Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["household_power"], name="household_power",line={'color':'#5c5cd6'}),
            row=1, col=1, secondary_y=False) """
        
        
        Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.houseRW_load, line={'color':'orange'}, name="houseRW_load", stackgroup="power"),
            row=1, col=1, secondary_y=False)
        Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["ev_power"], line={'color':'rgb(77, 218, 193)'}, name="ev_power", stackgroup="power"),
            row=1, col=1, secondary_y=False)
        Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["total_load"], line={'color':'#9467bd'}, name="total_load"),
            row=1, col=1, secondary_y=False)

        # Down
        Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["overcost"], line={'color':'rgb(210, 80, 75)'}, name="grid-cost", stackgroup="cost"),
            row=1, col=1, secondary_y=True)

        Elvis_fig['layout']['yaxis1'].update(title='kW')
        Elvis_fig['layout']['yaxis2'].update(title='€')
        Elvis_fig['layout']['legend'].update(title='Time series')
        Elvis_fig.update_layout(title_text='Elvis Load, EVs power, Grid-cost', width=1500,height= 600, xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.05, xaxis_range=["2022-06-01 00:00:00", "2022-07-01 00:00:00"])
        #Elvis_fig.show()
        return Elvis_fig

    def plot_VPP_results(self):
        """
        Method to plot and visualize the VPP simulation results (Input dataset superimposed with load, EV power and overcost)
        with charging actions controlled by the RL agent.
        """
        #Optimized VPP simulation graphs
        VPP_opt_fig = make_subplots(rows=1, cols=1,
                                    #shared_xaxes=True,
                                    specs=[[{"secondary_y": True}]])

        #VPP_opt_fig.add_trace(
        #    go.Scatter(x=self.elvis_time_serie, y=[0]*self.tot_simulation_len,line={'color':'#00174f'}, name="zero_load"),
        #    row=1, col=1, secondary_y=False)
        """ VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_data["solar_power"], name="solar_power",line={'color':'#95bf00'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_data["wind_power"], name="wind_power",line={'color':'#1ac6ff'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["household_power"], name="household_power",line={'color':'#5c5cd6'}),
            row=1, col=1, secondary_y=False) """
            

        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.houseRW_load, line={'color':'orange'}, name="houseRW_load", stackgroup="power"),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.optimized_VPP_data["ev_power"], line={'color':'rgb(77, 218, 193)'}, name="ev_power", stackgroup="power"),
            row=1, col=1, secondary_y=False)

        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.optimized_VPP_data["total_load"], line={'color':'#9467bd'}, name="total_load"),
            row=1, col=1, secondary_y=False)
        # Down
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.optimized_VPP_data["overcost"],line={'color':'rgb(210, 80, 75)'}, name="grid-cost", stackgroup="cost"),
            row=1, col=1, secondary_y=True)

        VPP_opt_fig['layout']['yaxis1'].update(title='kW')
        #VPP_opt_fig['layout']['yaxis2'].update(title='kW')
        VPP_opt_fig['layout']['yaxis2'].update(title='€')
        VPP_opt_fig['layout']['legend'].update(title='Time series')
        VPP_opt_fig.update_layout(title_text='VPP Load, EVs power, Grid-cost', width=1500,height= 600, xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.05, xaxis_range=["2022-06-01 00:00:00", "2022-07-01 00:00:00"])
        #VPP_opt_fig.show()
        return VPP_opt_fig
    
    def plot_VPP_supply_demand(self):
        """
        Method to plot and visualize the VPP supply/demand energy over time (Input dataset superimposed with charging/discharging EV power, resulting total load)
        with charging actions controlled by the RL agent.
        Update plot: supply/demand usage for Self-consumption/autarky analysis
        """
        #Optimized VPP simulation graphs
        VPP_opt_fig = make_subplots(#subplot_titles=(f'Supply/demand sources', f'Supply/demand usage'), shared_xaxes=True,
                                    rows=1, cols=1, 
                                    specs=[[{"secondary_y": False}],
                                            #[{"secondary_y": False}]
                                            ])
        #UP
        #Households consumption power sources
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["house_self-consump."], name="RE2house_self-consump.", stackgroup='positive',line={'color':'#67ff24'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["battery-self-consump."], name="EV2house_self-consump.", stackgroup='positive',line={'color':'#fc24ff'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["house-grid-import"], name="Grid2house_import", stackgroup='positive'),
            row=1, col=1, secondary_y=False)
        #EV charging power sources
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["self_EV-charging"], name="RE2EV_self-consump.", stackgroup='positive',line={'color':'#24f3ff'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["self_battery-EV-charging"], name="EV2EV_self-consump.", stackgroup='positive',line={'color':'#fff824'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["grid-import_EV-charging"], name="Grid2EV_import", stackgroup='positive'),
            row=1, col=1, secondary_y=False)
        #Consumption entities
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["household_power"], name="CO_household_power",line={'color':'#5c5cd6'}, stackgroup='consumed'),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["charging_ev_power"],line={'color':'#45d3d3'}, name="CO_EV_charging_pwr", stackgroup='consumed'),
            row=1, col=1, secondary_y=False)

        # DOWN
        #Renewable produced power usage
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["house_self-consump."], name="RE2house_self-consump.", stackgroup='negative',line={'color':'#67ff24'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["self_EV-charging"], name="RE2EV_self-consump.", stackgroup='negative',line={'color':'#24f3ff'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["RE-grid-export"], name="RE2grid_export", stackgroup='negative'),
            row=1, col=1, secondary_y=False)
        #EV discharged power usage
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["battery-self-consump."], name="EV2house_self-consump.", stackgroup='negative',line={'color':'#fc24ff'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["self_battery-EV-charging"], name="EV2EV_self-consump.", stackgroup='negative',line={'color':'#fff824'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["battery-grid-export"], name="EV2grid_export", stackgroup='negative'),
            row=1, col=1, secondary_y=False)
        #Production sources
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["solar_power"], name="PRO_solar_power",line={'color':'#95bf00'}, stackgroup='produced'),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["wind_power"], name="PRO_wind_power",line={'color':'#1ac6ff'}, stackgroup='produced'),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["RE_power"], name="PRO_RE_power",line={'color':'rgb(45, 167, 176)'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["discharging_ev_power"],line={'color':'#fa1d9c'}, name="PRO_ev_discharged_pwr", stackgroup='produced'),
            row=1, col=1, secondary_y=False)


        VPP_opt_fig['layout']['yaxis1'].update(title='kW')
        #VPP_opt_fig['layout']['yaxis2'].update(title='kW')
        VPP_opt_fig['layout']['legend'].update(title='Time series')
        VPP_opt_fig.update_layout(width=1500,height= 750,
                                    #barmode='stack', 
                                    title_text='VPP Supply/demand power',
                                    xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.05, xaxis_range=["2022-06-01 00:00:00", "2022-06-10 00:00:00"])
        #VPP_opt_fig.show()
        return VPP_opt_fig

    def plot_rewards_results(self):
        """
        Method to plot and visualize the rewards (total load for every step, EVs energy left at departure)
        over time during the VPP simulation with controlled charging actions.
        """
        rewards_fig = make_subplots(subplot_titles=('Load reward over time','EVs reward over time'),
                            rows=2, cols=1, shared_xaxes=True,
                            specs=[[{"secondary_y": True}],
                                    [{"secondary_y": True}]])
        
        rewards_serie = self.optimized_VPP_data["rewards"].values
        rewards_serie[-2] = 0
        # Top graph
        rewards_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.optimized_VPP_data["total_load"], line={'color':'#9467bd'}, name="total_load", stackgroup='load'),
            row=1, col=1, secondary_y=False)
        rewards_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.load_reward_hist, line={'color':'rgb(45, 167, 176)'}, name="load_rewards", stackgroup='reward'),
            row=1, col=1, secondary_y=True)
        
        rewards_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.avail_EVs_n, line={'color':'rgb(238, 173, 81)'}, name="EVs_available", stackgroup='evs'),
            row=2, col=1, secondary_y=False)
        rewards_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=rewards_serie, line={'color':'rgb(115, 212, 127)'}, name="total_reward", stackgroup='tot_reward'),
            row=2, col=1, secondary_y=True)
        rewards_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.EVs_reward_hist, line={'color':'rgb(210, 80, 75)'}, name="EVs_rewards"),
            row=2, col=1, secondary_y=True)

        self.load_reward_hist
        self.VPP_table["EV_reward"] = self.EVs_reward_hist
        self.VPP_table["rewards"] = self.reward_hist
        rewards_fig['layout']['yaxis1'].update(title='kW')
        rewards_fig['layout']['yaxis2'].update(title='Score')
        rewards_fig['layout']['yaxis3'].update(title='n_EVs')
        rewards_fig['layout']['yaxis4'].update(title='Score')
        
        rewards_fig.update_layout(title_text='Rewards results', width=1500,height=600, xaxis2_rangeslider_visible=True, xaxis2_rangeslider_thickness=0.05, xaxis_range=["2022-06-01 00:00:00", "2022-07-01 00:00:00"])
        return rewards_fig

    def plot_rewards_stats(self):
        """
        Method to plot and visualize the VPP agent cumulative reward and each reward instance composing it divided per category:
        - cumulative: the sum of all reward instances (final-rewards and step-rewards)
        - final_total: the sum of the final-reward instances
        - step_total: the sum of the step-reward instances
        - step_EV_en: the sum of all the rewards given at timesteps t when an EV left the charging station according to the energy left
        - step_load: the sum of all the rewards given at each timestep according the total load
        - final_Av_EV_en: the final reward evaluated for the Average EVs energy left when leaving the charging station
        - final_over_en: the final reward evaluated for the total energy consumed from the grid (not autosufficient)
        - final_under_en: the final reward evaluated for the total energy wasted (produced but not used)
        - final_overcost: the final reward evaluated for the total energy consumed price (energy bought)
        """
        rewards_fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": False}]])

        final_reward = (self.AV_EVs_energy_reward + self.overconsume_reward + self.underconsume_reward + self.overcost_reward)
        step_reward = (self.EVs_energy_reward + self.load_t_reward)
        rewards_fig.add_trace(go.Bar(x=["cumulative", 'final_total', "step_total", "step_EV_en", "step_load", "final_Av_EV_en", "final_Grid_en",  "final_RE-to-EV_unused_en","final_Grid-cost"],
                            y=[self.cumulative_reward, final_reward, step_reward, self.EVs_energy_reward, self.load_t_reward, self.AV_EVs_energy_reward, self.overconsume_reward, self.underconsume_reward, self.overcost_reward],
                            marker_color=['rgb(117, 122, 178)', 'rgb(156, 99, 255)', 'rgb(115, 212, 127)', 'rgb(210, 80, 75)', 'rgb(45, 167, 176)', 'rgb(238, 173, 81)', 'rgb(249, 152, 179)', 'rgb(77, 218, 193)', 'rgb(97, 159, 210)']),
                            row=1, col=1, secondary_y=False)
        
        #rewards_fig['layout']['yaxis1'].update(title='Score')
        rewards_fig['layout']['yaxis1'].update(title='Score')
        #rewards_fig['layout']['legend'].update(title=f'Cumulat.Reward= {round(self.cumulative_reward,2)}')
        rewards_fig.update_layout(title_text="Cumulative, Step, Final reward bars comparison", width=1500,height=500,)
        #rewards_fig.show()
        return rewards_fig
    
    def plot_VPP_Elvis_comparison(self):
        """
        Method to plot and visualize with bars the VPP simulation with controlled charging results compared to the ELVIS uncontrolled charging ones.
        """
        comparison_fig = make_subplots(subplot_titles=('Av.EVs energy at departure','Grid used en.','RE-to-vehicle unused en.', 'Grid-cost'),
                            rows=1, cols=4,
                            specs=[[{"secondary_y": False}, {"secondary_y": False},{"secondary_y": False},{"secondary_y": False}]])

        x = ["Elvis_simulation","VPP_simulation"]
        marker_color = ['#636efa', 'rgb(77, 218, 193)']
        comparison_fig.add_trace(go.Bar(x=["Elvis_simulation","VPP_simulation","Expected"], y=[self.Elvis_av_EV_energy_left, self.av_EV_energy_left, self.exp_ev_en_left], marker_color=['#636efa', 'rgb(77, 218, 193)','orange']),row=1, col=1)
        comparison_fig.add_trace(go.Bar(x=x, y=[self.Elvis_overconsume, self.overconsumed_en], marker_color=marker_color),row=1, col=2)
        comparison_fig.add_trace(go.Bar(x=x, y=[self.Elvis_underconsume, self.underconsumed_en], marker_color=marker_color),row=1, col=3)
        comparison_fig.add_trace(go.Bar(x=x, y=[self.Elvis_overcost, self.sim_overcost], marker_color=marker_color),row=1, col=4)
        comparison_fig['layout']['yaxis1'].update(title='kWh')
        comparison_fig['layout']['yaxis2'].update(title='kWh')
        comparison_fig['layout']['yaxis3'].update(title='kWh')
        comparison_fig['layout']['yaxis4'].update(title='€')
        comparison_fig.update_layout(title_text='VPP/Elvis simulation comparison', width=1500,height=500, showlegend = False)
        #comparison_fig.show()
        return comparison_fig

    def plot_EVs_kpi(self):
        """
        Method to plot and visualize the histogram of the EVs energy left at departure during the VPP simulation with controlled charging.
        """
        kpi_fig = px.histogram(x=self.EVs_energy_at_leaving, marginal = 'violin')
        kpi_fig.update_xaxes(title = 'energy% left (kWh)')
        kpi_fig.update_layout(title_text="EVs energy at departure histogram",  width=1500,height=700,)
        #kpi_fig.show()
        return kpi_fig
    
    def plot_actions_kpi(self):
        """
        Method to plot and visualize the heatmap of the valid actions taken by the agent during the VPP simulation with controlled charging.
        """
        kpi_fig = make_subplots(subplot_titles=("Valid actions per station table","Vehicle availability per station table"),
                            rows=1, cols=2,
                            specs=[[{"secondary_y": False},{"secondary_y": False}]])

        kpi_fig.add_trace(go.Heatmap(z=self.action_truth_table.astype(int), colorscale='Viridis', colorbar_x=0.45), row=1, col=1)
        #self.Evs_id_table = self.Evs_id_table.astype(bool)
        kpi_fig.add_trace(go.Heatmap(z=self.Evs_id_table), row=1, col=2)
        kpi_fig.update_layout(title_text='Actions KPIs', width=1500,height=500,)
        kpi_fig['layout']['yaxis1'].update(title='timesteps')
        kpi_fig['layout']['yaxis2'].update(title='timesteps')
        kpi_fig['layout']['xaxis1'].update(title='ch.station number')
        kpi_fig['layout']['xaxis2'].update(title='ch.station number')
        #kpi_fig.show()
        return kpi_fig

    def plot_load_kpi(self):
        """
        Method to plot and visualize the histogram of the timesteps load values during the ELVIS uncontrolled charging simulation,
        and during the VPP simulation with controlled charging (Weekly, Monthly, Yearly)
        """
        kpi_fig = make_subplots(subplot_titles=('Elvis Weekly',"Elvis Monthly","Elvis Yearly",'VPP Weekly',"VPP Monthly","VPP Yearly"), rows=2, cols=3,
                            specs=[[{"secondary_y": False},{"secondary_y": False},{"secondary_y": False}],
                                    [{"secondary_y": False},{"secondary_y": False},{"secondary_y": False}]])

        kpi_fig.add_trace(go.Histogram(x=self.VPP_data["total_load"].loc["2022-01-01 00:00:00":"2022-01-08 00:00:00"], marker = dict(color ='#7663fa')), row=1, col=1)
        kpi_fig.add_trace(go.Histogram(x=self.VPP_data["total_load"].loc["2022-06-01 00:00:00":"2022-07-01 00:00:00"], marker = dict(color ='#636efa')), row=1, col=2)
        kpi_fig.add_trace(go.Histogram(x=self.VPP_data["total_load"], marker = dict(color ='#636efa')), row=1, col=3)
        kpi_fig.add_trace(go.Histogram(x=self.optimized_VPP_data["total_load"].loc["2022-01-01 00:00:00":"2022-01-08 00:00:00"], marker = dict(color ='#00cc96')), row=2, col=1)
        kpi_fig.add_trace(go.Histogram(x=self.optimized_VPP_data["total_load"].loc["2022-06-01 00:00:00":"2022-07-01 00:00:00"], marker = dict(color ='#00cc96')), row=2, col=2)
        kpi_fig.add_trace(go.Histogram(x=self.optimized_VPP_data["total_load"], marker = dict(color ='rgb(77, 218, 193)')), row=2, col=3)
        
        kpi_fig['layout']['xaxis1'].update(title='kW')
        kpi_fig['layout']['xaxis2'].update(title='kW')
        kpi_fig['layout']['xaxis3'].update(title='kW')
        kpi_fig['layout']['xaxis4'].update(title='kW')
        kpi_fig['layout']['xaxis5'].update(title='kW')
        kpi_fig['layout']['xaxis6'].update(title='kW')
        kpi_fig['layout']['yaxis1'].update(title='load value occurences')
        kpi_fig['layout']['yaxis2'].update(title='load value occurences')
        kpi_fig['layout']['yaxis3'].update(title='load value occurences')
        kpi_fig['layout']['yaxis4'].update(title='load value occurences')
        kpi_fig['layout']['yaxis5'].update(title='load value occurences')
        kpi_fig['layout']['yaxis6'].update(title='load value occurences')
        kpi_fig.update_layout(title_text='Load peak occurences histograms',  width=1500,height=800, showlegend = False)
        #kpi_fig.show()
        return kpi_fig

    
    def plot_yearly_load_log(self):
        """
        Method to plot and visualize the logaritmic histogram of the timesteps load values during the ELVIS uncontrolled charging simulation,
        during the VPP simulation with controlled charging (Yearly, superimposed)
        """
        x0 = [0]*self.tot_simulation_len
        x1 = self.optimized_VPP_data["total_load"].values
        x2 = self.VPP_data["total_load"].values
        df =pd.DataFrame(dict(series = np.concatenate((["steady-zero-load"]*len(x0), ["VPP-load"]*len(x1),  ["ELVIS-load"]*len(x2))), 
                                kW  = np.concatenate((x0,x1,x2))
                            ))

        kpi_fig = px.histogram(df, x="kW", color="series", barmode="overlay", marginal = 'violin', log_y=True, color_discrete_map = {"steady-zero-load":'orange', "VPP-load":'rgb(77, 218, 193)', "ELVIS-load":'#636efa'})
        kpi_fig.update_layout(title_text='Yearly Load peak occurences histogram',  width=1500,height=700,)
        #kpi_fig.show()
        return kpi_fig