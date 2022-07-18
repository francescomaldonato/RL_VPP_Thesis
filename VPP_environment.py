import os
import contextlib
from gym import Env
from gym.spaces import Box, Dict, MultiDiscrete
import numpy as np
#import pandas as pd

from elvis.simulate import simulate
from elvis.utility.elvis_general import create_time_steps
from elvis.utility.elvis_general import num_time_steps


class VPPEnv(Env):
  

    def __init__(self, elvis_config_file, VPP_data, simulation_param):
        
        #Costants for all episodes:
        self.elvis_config_file = elvis_config_file
        self.start = simulation_param["start_date"]
        self.end = simulation_param["end_date"]
        self.res = simulation_param["resolution"]
        self.charging_stations_n = simulation_param["charging_stations_n"]
        self.EVs_n_max = simulation_param["EVs_n_max"]
        self.EV_load_max = simulation_param["EV_load_max"]
        self.houseRWload_max = simulation_param["houseRWload_max"]

        elvis_realisation = elvis_config_file.create_realisation(self.start, self.end, self.res)

        #ELVIS Initial simulation
        #self.realisation = elvis_realisation
        #To be updated each episode:
        self.charging_events = elvis_realisation.charging_events

        """ for j in range(len(self.charging_events)):
            print(self.charging_events[j], '\n') """
        print(self.charging_events[0], '\n', '...', '\n', self.charging_events[-1], '\n')

        self.current_charging_events = []
        self.simul_charging_events_n = len(self.charging_events)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            result = simulate(elvis_realisation)
        VPP_data["ev_power"] = result.aggregate_load_profile(num_time_steps(elvis_realisation.start_date, elvis_realisation.end_date, elvis_realisation.resolution))
        #VPP_data["ev_power"].plot()
        VPP_data["total_load"] = VPP_data["House&RW_load"] + VPP_data["ev_power"]
        VPP_data["total_cost"] = VPP_data["total_load"] * VPP_data["EUR/kWh"] / 4
        self.Elvis_total_load = np.sum(VPP_data["total_load"].values/4) #kWh
        self.Elvis_total_cost = np.sum(VPP_data["total_cost"].values) #€
        self.Elvis_av_ev_load = np.mean(VPP_data["ev_power"].values) #kW

        print("Elvis simulation: Total energy consumed= KWh ", self.Elvis_total_load,
                  ", Total cost= € ", self.Elvis_total_cost,
                  ", Av. EV load= ", self.Elvis_av_ev_load,
                  ", Charging events= ", self.simul_charging_events_n)

        #Set VPP session length
        self.elvis_time_serie = create_time_steps(elvis_realisation.start_date, elvis_realisation.end_date, elvis_realisation.resolution)
        self.tot_simulation_len = len(self.elvis_time_serie)
        self.vpp_length = self.tot_simulation_len

        self.energy_resources, self.avail_EVs_id, self.ev_power, self.total_cost, self.total_load = ([],[],[],[],[])
        
        self.prices_serie = VPP_data["EUR/kWh"].values #EUR/kWh
        self.max_price_serie = self.prices_serie[np.argmax(self.prices_serie)]
        self.houseRW_load = VPP_data["House&RW_load"].values
        self.max_total_load = self.EV_load_max+self.houseRWload_max
        self.max_cost = self.max_total_load * self.max_price_serie /4

        self.VPP_data = VPP_data

        self.VPP_actions = []
        self.EVs_energy_at_leaving = []
        
        #Initializing state space parameters for the first step [0]
        Init_space_Available_energy_sources = np.zeros(self.charging_stations_n, dtype=np.float32)
        Init_space_Available_EVs_id = np.zeros(self.charging_stations_n, dtype=np.int32)

        Init_space_ev_power = np.zeros(1,dtype=np.float32)
        Init_space_total_load = np.zeros(1,dtype=np.float32)
        Init_space_total_cost = np.zeros(1,dtype=np.float32)
        Init_space_total_load[0] = self.houseRW_load[0]
        Init_space_total_cost[0] = self.houseRW_load[0] * self.prices_serie[0]/4

        self.Init_space = {
            'ev_power': Init_space_ev_power, #EV load range in kW, depends on the infrastructure of the simulation
            'total_load': Init_space_total_load, #Total load range in kW, depends on the household and EV infrastructure of the simulation
            'total_cost': Init_space_total_cost, #Total cost (EUR) per timestep (15 min)

            'Available_energy_sources': Init_space_Available_energy_sources, #Available energy from Charging stations (EVs connected)
            'Available_evs_id': Init_space_Available_EVs_id #Available vehicles IDs from charging stations at timestep t
            }
        
        #For plotting battery levels
        self.VPP_energies = Init_space_Available_energy_sources
        
        # Define constants for Action space options
        self.IDLE = 0
        self.CHARGE_3_7 = 1
        self.DISCHARGE_3_7 = 2

        self.CHARGE_7_4 = 3
        self.DISCHARGE_7_4 = 4

        self.CHARGE_11 = 5
        self.DISCHARGE_11 = 6

        #Action space definition
        self.action_space = MultiDiscrete( 7 * np.ones(self.charging_stations_n))

        #TO DO: NORMALIZE ACTION SPACE (not if you use algorithm TD3 and DDPG)
        spaces = {
            'ev_power': Box(low=-(self.EV_load_max), high=(self.EV_load_max), shape=(1,), dtype=np.float32), #EV load range in kW, depends on the infrastructure of the simulation
            'total_load': Box(low=-(self.max_total_load) , high= (self.max_total_load), shape=(1,), dtype=np.float32), #Total load range in kW, depends on the household and EV infrastructure of the simulation
            'total_cost': Box(low=-(self.max_cost), high=(self.max_cost), shape=(1,), dtype=np.float32),#Total cost (EUR) per timestep (15 min)

            'Available_energy_sources': Box(low=0.0, high=100, shape=(self.charging_stations_n,), dtype=np.float32),
            'Available_evs_id': Box(low=0, high=(np.iinfo(np.int32).max)-1, shape=(self.charging_stations_n,), dtype=np.int32)
            #'Available_evs_id': MultiDiscrete( (self.EVs_n_max) * np.ones(self.charging_stations_n), dtype=np.int32)
            }

        dict_space = Dict(spaces)
        self.observation_space = dict_space
        #Set starting cond.
        self.state = self.Init_space
        
        self.done = False


    def eval_reward(self, reward, step, new_ev_departures):
        
        #Load step state variables
        total_ev_power_t = self.ev_power[step]
        cost_t = self.total_cost[step]
        total_load_t = self.total_load[step]

        #Apply reward on energy left on vehicle leaving the station 
        for j in range(new_ev_departures):
            energy_left = self.EVs_energy_at_leaving[-1-j]

            if 0 < energy_left <= 25:
                reward += -100
            elif 25 < energy_left <= 40:
                reward += -50
            elif 40 < energy_left <= 55:
                reward += -10
            elif 55 < energy_left <= 65:
                reward += 0
            elif 65 < energy_left <= 85:
                reward += 50
            elif 85 < energy_left <= 95:
                reward += 100
            elif 85 < energy_left <= 99:
                reward += 150
            elif 99 < energy_left <= 100:
                reward += 50
        
        #Evaluating rewards for EV_power, total_load, and cost at instant t
        #EV power reward
        total_ev_power_t = (total_ev_power_t / self.EV_load_max) * 100 #Normalizing EV power, percentage
        if total_ev_power_t <= -20 :
            reward += 20
        elif -20 < total_ev_power_t <= -10:
            reward += 30
        elif -10 < total_ev_power_t <= -1:
            reward += 20
        elif -1 < total_ev_power_t <= 1:
            reward += 10
        elif 1 < total_ev_power_t <= 10:
            reward += 0
        elif 10 < total_ev_power_t <= 20:
            reward += -5
        elif 20 < total_ev_power_t <= 40:
            reward -= 15
        elif 40 < total_ev_power_t <= 60:
            reward -= 25
        elif 60 < total_ev_power_t <= 80:
            reward -= 35
        elif 80 < total_ev_power_t <= 100:
            reward -= 45
        elif total_ev_power_t  > 100:
            reward -= 50

        #Total load reward
        total_load_t = (total_load_t / self.max_total_load) * 100 #Normalizing Total load
        if total_load_t <= -20 :
            reward += 20
        elif -20 < total_load_t <= -10:
            reward += 30
        elif -10 < total_load_t <= -1:
            reward += 20
        elif -1 < total_load_t <= 1:
            reward += 10
        elif 1 < total_load_t <= 10:
            reward += 0
        elif 10 < total_load_t <= 20:
            reward += -5
        elif 20 < total_load_t <= 40:
            reward -= 15
        elif 40 < total_load_t <= 60:
            reward -= 25
        elif 60 < total_load_t <= 80:
            reward -= 35
        elif 80 < total_load_t <= 100:
            reward -= 45
        elif total_load_t  > 100:
            reward -= 50
        
        #Cost reward
        cost_t = (cost_t / self.max_cost) * 100 #Normalizing cost
        if cost_t <= -20 :
            reward += 20
        elif -20 < cost_t <= -10:
            reward += 30
        elif -10 < cost_t <= -1:
            reward += 20
        elif -1 < cost_t <= 1:
            reward += 10
        elif 1 < cost_t <= 10:
            reward += 0
        elif 10 < cost_t <= 20:
            reward += -5
        elif 20 < cost_t <= 40:
            reward -= 15
        elif 40 < cost_t <= 60:
            reward -= 25
        elif 60 < cost_t <= 80:
            reward -= 35
        elif 80 < cost_t <= 100:
            reward -= 45
        elif cost_t  > 100:
            reward -= 50

        return reward
    
    def eval_final_reward(self, reward, sim_total_load, total_cost, av_energy_left):
        #total_load_t,total_cost_t,av_energy_left, av_ev_load
        #Sum of Total load up to instant t reward
        sim_total_load = (sim_total_load / self.Elvis_total_load) * 100 #Normalizing simulation total load
        if sim_total_load <= -20 :
            reward += 4000
        elif -20 < sim_total_load <= -10:
            reward += 3000
        elif -10 < sim_total_load <= -1:
            reward += 2000
        elif -1 < sim_total_load <= 1:
            reward += 1000
        elif 1 < sim_total_load <= 10:
            reward += 100
        elif 10 < sim_total_load <= 20:
            reward += -500
        elif 20 < sim_total_load <= 40:
            reward -= 1500
        elif 40 < sim_total_load <= 60:
            reward -= 2500
        elif 60 < sim_total_load <= 80:
            reward -= 3500
        elif 80 < sim_total_load <= 100:
            reward -= 4500
        elif sim_total_load  > 100:
            reward -= 5000

        #Sum of Total Cost up to instant t reward
        total_cost = (total_cost / self.Elvis_total_cost) * 100
        if total_cost <= -20 :
            reward += 4000
        elif -20 < total_cost <= -10:
            reward += 3000
        elif -10 < total_cost <= -1:
            reward += 2000
        elif -1 < total_cost <= 1:
            reward += 1000
        elif 1 < total_cost <= 10:
            reward += 100
        elif 10 < total_cost <= 20:
            reward += -500
        elif 20 < total_cost <= 40:
            reward -= 1500
        elif 40 < total_cost <= 60:
            reward -= 2500
        elif 60 < total_cost <= 80:
            reward -= 3500
        elif 80 < total_cost <= 100:
            reward -= 4500
        elif total_cost  > 100:
            reward -= 5000

        #Evaluating reward for average energy left in EV leaving
        if 0 <= av_energy_left <= 25:
            reward += -10000
        elif 25 < av_energy_left <= 40:
            reward += -5000
        elif 40 < av_energy_left <= 55:
            reward += -1000
        elif 55 < av_energy_left <= 65:
            reward += 100
        elif 65 < av_energy_left <= 85:
            reward += 5000
        elif 85 < av_energy_left <= 95:
            reward += 10000
        elif 85 < av_energy_left <= 99:
            reward += 20000
        elif 99 < av_energy_left <= 100:
            reward += 50000

        return reward
    
    def apply_action_on_energy_source(self, Energy_sources_t_1, action, total_ev_power_t):

        #APPLY ACTION on previous energy state:
        if action == self.CHARGE_3_7:
            Energy_sources_t = Energy_sources_t_1 + (3.7 * 0.25) #3.7 kW * 15 min = kWh STORING ENERGY
            total_ev_power_t += 3.7
            if Energy_sources_t > 100: #Reached max capacity (kWh)
                total_ev_power_t -= 3.7
                Energy_sources_t = 100
        elif action == self.CHARGE_7_4:
            Energy_sources_t = Energy_sources_t_1 + (7.4 * 0.25) #7.4 kW * 15 min = kWh STORING ENERGY
            total_ev_power_t += 7.4
            if Energy_sources_t > 100: #Reached max capacity (kWh)
                total_ev_power_t -= 7.4
                Energy_sources_t = 100
        elif action == self.CHARGE_11:
            Energy_sources_t = Energy_sources_t_1 + (11 * 0.25) #11 kW * 15 min = kWh STORING ENERGY
            total_ev_power_t += 11
            if Energy_sources_t > 100: #Reached max capacity (kWh)
                total_ev_power_t -= 11
                Energy_sources_t = 100
        elif action == self.IDLE:
            Energy_sources_t = Energy_sources_t_1 #keep energy constant
        elif action == self.DISCHARGE_3_7:
            Energy_sources_t = Energy_sources_t_1 - (3.7 * 0.25) #3.7 kW * 15 min = kWh PUSHING ENERGY
            total_ev_power_t -= 3.7
            if Energy_sources_t < 0.01: #Reached min capacity (kWh)
                total_ev_power_t += 3.7
                Energy_sources_t = 0.01
        elif action == self.DISCHARGE_7_4:
            Energy_sources_t = Energy_sources_t_1 - (7.4 * 0.25) #7.4 kW * 15 min = kWh PUSHING ENERGY
            total_ev_power_t -= 7.4
            if Energy_sources_t < 0.01: #Reached min capacity (kWh)
                total_ev_power_t += 7.4
                Energy_sources_t = 0.01
        elif action == self.DISCHARGE_11:
            Energy_sources_t = Energy_sources_t_1 - (11 * 0.25) #3.7 kW * 15 min = kWh PUSHING ENERGY
            total_ev_power_t -= 11
            if Energy_sources_t < 0.01: #Reached min capacity (kWh)
                total_ev_power_t += 11
                Energy_sources_t = 0.01
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        return Energy_sources_t, total_ev_power_t


    def step(self, action):
        #Reduce VPP session length by 1 step [Episode init and reset do as step 0, first step = 1]
        self.vpp_length -= 1
        
        #evaluate step
        step = self.tot_simulation_len - self.vpp_length
        time_step = self.elvis_time_serie[step]
    
        #loding step variables
        Evs_id_t = self.avail_EVs_id[step]
        Energy_sources_t = self.energy_resources[step]
        Evs_id_t_1 = self.avail_EVs_id[step-1]
        Energy_sources_t_1 = self.energy_resources[step-1]

        #EVs_energy_at_leaving = self.EVs_energy_at_leaving

        #Reward Initialization
        reward = 0
        #Variables inititalization for calculations
        total_ev_power_t = 0

        new_ev_departures = 0

        #SECTION 1. Check if current connected EVs left the charging station
        for charging_event in self.current_charging_events:
            leaving_time_i = charging_event.leaving_time
            if time_step >= leaving_time_i:
                #If vehicle left, set correspondant station ID to zero 
                n = charging_event.station_n
                energy_at_leaving_i, total_ev_power_t = self.apply_action_on_energy_source(Energy_sources_t_1[n], action[n], total_ev_power_t)
                self.EVs_energy_at_leaving.append(energy_at_leaving_i)
                new_ev_departures += 1
                Evs_id_t[n] = int(0)
                self.current_charging_events.remove(charging_event)
            else:
                #If Vehicle still connected, correspondant station ID = EV's ID
                Evs_id_t[charging_event.station_n] = charging_event.id
            #__END__ SECTION 1

        #SECTION 2. Apply action to each station section 
        for n in range(self.charging_stations_n):
            #1. Check Evs id present and evaluate new Energy available at station n
            if Evs_id_t[n] > 0:
                Energy_sources_t[n], total_ev_power_t = self.apply_action_on_energy_source(Energy_sources_t_1[n], action[n], total_ev_power_t)

            elif Evs_id_t[n] == 0:
                #If no car is connected at station n, available energy = 0
                if Energy_sources_t[n] != 0:
                    raise ValueError("Available_energy_sources table not matching EVs id: state={} where there is an empty station with a certain energy.".format(Energy_sources_t))
            
            if Evs_id_t_1[n] == 0:
                #Punishment for not giving IDLE action to an empty station
                if action[n] != 0:
                    reward += -100

            if Energy_sources_t[n] < 0 or Energy_sources_t[n] > 100:
                #Check if energy sources are out of range
                raise ValueError("Available_energy_sources table out of ranges: state={} which is not part of the state space".format(Energy_sources_t))
            #__END__ SECTION 2


        #SECTION 3. Check if new vehicles arrive at charging stations
        for charging_event in self.charging_events:
            arrival_time_i = charging_event.arrival_time
            #Fixing arrival time at step 0, shifted to step 1
            if step == 1:
                if arrival_time_i == self.elvis_time_serie[0]:
                    arrival_time_i = self.elvis_time_serie[1]
            if arrival_time_i == time_step:
                vehicle_id = int(charging_event.id[16:])
                charging_event.id = vehicle_id
                for n in range(self.charging_stations_n):
                        #Check for free station, when found overwrite data and exit loop
                        if Evs_id_t[n] == 0:
                            Evs_id_t[n] = vehicle_id

                            vehicle_i = charging_event.vehicle_type.to_dict()
                            soc_i = charging_event.soc
                            battery_i = vehicle_i['battery']
                            #efficiency_i  = battery_i['efficiency'] #Not implemented right now
                            capacity_i  = battery_i['capacity'] #kWh
                            #capacity_i  = 100 #kWh, considering only Tesla Model S
                            energy_i = soc_i * capacity_i #kWh

                            Energy_sources_t[n] = energy_i
                            charging_event.station_n = n
                            self.current_charging_events.append(charging_event)
                            self.charging_events.remove(charging_event)
                            break
            elif arrival_time_i > time_step:
                break
            #__END__ SECTION 3
             

        #SECTION 4. VPP States and Values updates
        self.ev_power[step] = total_ev_power_t
        self.total_load[step] = self.houseRW_load[step] + total_ev_power_t
        self.total_cost[step] = self.total_load[step] * self.prices_serie[step] / 4
        self.avail_EVs_id[step] = Evs_id_t
        self.energy_resources[step] = Energy_sources_t
        #Evaluate step reward
        reward = self.eval_reward(reward, step, new_ev_departures)
        #VPP Table UPDATE
        self.VPP_energies = np.vstack((self.VPP_energies,Energy_sources_t))
        self.VPP_actions.append(action)

        #States UPDATE
        self.state['Available_energy_sources'] = Energy_sources_t
        self.state['Available_evs_id'] = Evs_id_t

        ev_power_state = np.zeros(1,dtype=np.float32)
        ev_power_state[0] = total_ev_power_t
        self.state['ev_power'] = ev_power_state
        load_state = np.zeros(1,dtype=np.float32)
        load_state[0] = self.total_load[step]
        self.state['total_load'] = load_state
        cost_state = np.zeros(1,dtype=np.float32)
        cost_state[0] = self.total_cost[step]
        self.state['total_cost'] = cost_state

        #FINAL_SECTION: Check if VPP is done
        if self.vpp_length <= 1:
            self.done = True
            self.VPP_actions.append(np.zeros(self.charging_stations_n, dtype=np.int32))

            #Evaluating sum and average up to timestep t for further rewards
            av_energy_left = np.mean(self.EVs_energy_at_leaving)
            total_load_t = np.sum(self.total_load)
            total_cost_t = np.sum(self.total_cost)
            av_ev_load = np.mean(self.ev_power)
            charging_events_n = len(self.EVs_energy_at_leaving)
            #Final reward
            reward = self.eval_final_reward(reward,total_load_t,total_cost_t,av_energy_left)

            print("VPP simulation:  Energy consumed=KWh ", round(total_load_t,4),
                  ", Total_cost=€ ", round(total_cost_t,2),
                  ", Av.EV_load=kW ", round(av_ev_load,4),
                  ", EV_leaving_stations= ", charging_events_n,
                  ", Av.EV_energy_leaving=kWh ", round(av_energy_left,2))
            #Save data
            #VPP_table_csv = VPP_table.to_csv(output_folder + 'VPP_table_complete.csv', index = True)
        else:
            self.done = False

        #set placeholder for info
        info = {}
        #return step information
        return self.state, reward, self.done, info
    
    
    def render(self, mode = 'human'):
        #implement visualization
        pass

    def reset(self):
        #reset VPP data
        elvis_config_file = self.elvis_config_file
        #Create new ELVIS simulation
        elvis_realisation = elvis_config_file.create_realisation(self.start, self.end, self.res)
        self.charging_events = elvis_realisation.charging_events
        self.current_charging_events = []
        self.simul_charging_events_n = len(self.charging_events)

        VPP_data = self.VPP_data
        #Apply noise on dataset:
        mu, sigma = 0, (self.max_price_serie/100)*7 #Mean, standard deviation
        noise = np.random.normal(mu, sigma, self.tot_simulation_len) # creating a noise with the same dimension of the dataset length
        VPP_data["EUR/kWh"] = VPP_data["EUR/kWh"].values + noise
        mu, sigma = 0, (self.houseRWload_max/100)*5 #Mean, standard deviation
        noise = np.random.normal(mu, sigma, self.tot_simulation_len) # creating a noise with the same dimension of the dataset length
        VPP_data["House&RW_load"] = VPP_data["House&RW_load"].values + noise
        #VPP_data["renewable_power"] = VPP_data["renewable_power"] + noise
        #VPP_data["household_power"] = VPP_data["household_power"] + noise
        #VPP_data["House&RW_load"] = VPP_data["household_power"] + VPP_data["renewable_power"]       

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            result = simulate(elvis_realisation)
        VPP_data["ev_power"] = result.aggregate_load_profile(num_time_steps(elvis_realisation.start_date, elvis_realisation.end_date, elvis_realisation.resolution))
        #VPP_data["ev_power"].plot()
        VPP_data["total_load"] = VPP_data["House&RW_load"] + VPP_data["ev_power"]
        VPP_data["total_cost"] = VPP_data["total_load"] * VPP_data["EUR/kWh"] / 4
        Elvis_total_load = np.sum(VPP_data["total_load"].values/4) #kWh
        Elvis_total_cost = np.sum(VPP_data["total_cost"].values) #€
        Elvis_av_ev_load = np.mean(VPP_data["ev_power"].values) #kW

        print("Elvis simulation: Energy_consumed=KWh ", round(Elvis_total_load,4),
                  ", Total_cost=€ ", round(Elvis_total_cost,2),
                  ", Av.EV_load=kW ", round(Elvis_av_ev_load,4),
                  ", Charging_events= ", self.simul_charging_events_n)

        #Reset VPP session length
        self.vpp_length = self.tot_simulation_len
        self.energy_resources, self.avail_EVs_id, self.ev_power, self.total_cost, self.total_load = ([],[],[],[],[])
        #build EV series (Avail_en. and IDs)
        for i in range(len(self.elvis_time_serie)):
            self.energy_resources.append(np.zeros(self.charging_stations_n, dtype=np.float32))
            self.avail_EVs_id.append(np.zeros(self.charging_stations_n, dtype=np.int32))
            self.ev_power.append(0.0)
            self.total_cost.append(0.0)
            self.total_load.append(0.0)
        
        self.prices_serie = VPP_data["EUR/kWh"].values #EUR/kWh
        self.houseRW_load = VPP_data["House&RW_load"].values
        self.total_load[0] = self.houseRW_load[0]
        self.total_cost[0] = self.total_load[0] * self.prices_serie[0]/4

        self.VPP_data = VPP_data

        self.VPP_actions = []
        self.EVs_energy_at_leaving = []

        #For plotting battery levels
        self.VPP_energies = self.Init_space["Available_energy_sources"]
        #Set starting cond.
        self.state = self.Init_space
        #reset vpp session time
        self.vpp_length = self.tot_simulation_len
        self.done = False
        return self.state
