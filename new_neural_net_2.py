#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from ns3gym import ns3env
from tcp_base import TcpTimeBased
from tcp_newreno import TcpNewReno

# Modification start
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
#Modification End

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"


parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=1,
                    help='Number of iterations, Default: 1')
args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)

port = 5555
simTime = 10 # seconds
stepTime = 0.5  # seconds
seed = 12
simArgs = {"--duration": simTime,}
debug = False

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
# simpler:
#env = ns3env.Ns3Env()
#env.reset()

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.dtype)

stepIdx = 0
currIt = 0

def get_agent(obs):
    socketUuid = obs[0]
    tcpEnvType = obs[1]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    if tcpAgent is None:
        if tcpEnvType == 0:
            # event-based = 0
            tcpAgent = TcpNewReno()
        else:
            # time-based = 1
            tcpAgent = TcpTimeBased()
            print("time-base")
        tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
        get_agent.tcpAgents[socketUuid] = tcpAgent

    return tcpAgent


GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000
BATCH_SIZE = 20
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNSolver:
            def __init__(self, observation_space, action_space):
                        self.exploration_rate = EXPLORATION_MAX

                        self.action_space = action_space
                        self.memory = deque(maxlen=MEMORY_SIZE)

                        self.model = Sequential()
                        self.model.add(Dense(24, input_shape = (observation_space,), activation="relu"))
                        self.model.add(Dense(24, activation="relu"))
                        self.model.add(Dense(self.action_space, activation="linear"))
                        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

            def remember(self, state, action, reward, next_state, done):
                self.memory.append((state, action, reward, next_state, done))

            def act(self, state):
                if np.random.rand() < self.exploration_rate:
                   return random.randrange(self.action_space)
                print(state.shape)
                q_values = self.model.predict(state)
                print("Q_Value: ",q_values)
                return np.argmax(q_values[0])

            def experience_replay(self):
                if len(self.memory) < BATCH_SIZE:
                    return
                batch = random.sample(self.memory, BATCH_SIZE)
                for state, action, reward, state_next, terminal in batch:
                    q_update = reward
                    if not terminal:
                        q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
                    q_values = self.model.predict(state)
                    q_values[0][action] = q_update
                    self.model.fit(state, q_values, verbose=0)
                self.exploration_rate *= EXPLORATION_DECAY
                self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

class neuralNet:

        def __init__(self,inc_acts=4,dec_acts=2):
                self.utility_prev1 = 0
                self.utility_prev = 0
                self.utility_now = 0
                self.del_utility_now = 0
                self.del_utility_prev = 0
                self.utility_change = 0
                self.no_switch = 1
                self.line_count = 1
                self.reward = 0
                self.done = [False, False]
                self.info = None
                self.flag = 1
                self.dqn_solver_inc = DQNSolver(2, inc_acts)
                self.dqn_solver_dec = DQNSolver(2, dec_acts)
                self.state = np.reshape([0,0],[1,2])
                self.index = -1
                
        def run(self,obs, Done=False):
          
          #storing the previous state done value in done[0] and present state in done[1]
                print("line_count", self.line_count)
                if(self.line_count %2 !=0):
                        self.done[0] = Done
                else:
                        self.done[1] = Done
                
          # Calclating Utility Functions
                if obs[11] == 0: rtt = np.e
                if obs[11] != 0: rtt = obs[11]*0.001
                if obs[15] == 0: thpt = np.e
                if obs[15] != 0: thpt = obs[15]

                self.utility_prev1 = self.utility_prev
                self.utility_prev = self.utility_now

                if thpt==np.e and rtt==np.e: wait = 1
                else: wait = 0
                if wait == 0:
                        self.utility_now = 0.6*np.log(thpt) - 0.4*np.log(rtt) 
                if wait == 1:
                        self.utility_now = 0

          # Printing utility values
                print("utility_now: ",self.utility_now)
                print("utility_prev: ",self.utility_prev)
                print("utility_prev1: ",self.utility_prev1)
                      
                self.del_utility_now = self.utility_now - self.utility_prev
                self.del_utility_prev = self.utility_prev - self.utility_prev1

                self.utility_change = self.del_utility_now - self.del_utility_prev
                
          # Printing change in utility values
                print("del_utility_now: ",self.del_utility_now)
                print("del_utility_prev: ",self.del_utility_prev)      
                print("utility_change: ",self.utility_change)
                
                if self.utility_change >= 0:
                        self.no_switch = 1
                        reward_prev = self.reward
                        self.reward = 10
                else:
                        self.no_switch = 0
                        reward_prev = self.reward
                        self.reward = -5
          # Printing flag and action before change
                print("Flag of prev state: ",self.flag)
                print("Node of present state:", obs[0])

           # Switching Operation between the actions
                if self.no_switch == 1:
                        if self.flag == 1: 
                                act = 1
                        else: 
                                act = 2
                if self.no_switch == 0:
                        if self.flag == 1: 
                                act = 2
                        if self.flag == 2: 
                                act = 1
                                
         # Printing action for present state
                print("Action for present state: ", act) 
                                     
           # Intialising the present state
                state_prev = self.state
                self.state = np.reshape([obs[13],obs[14]], [1,2])
                flag_prev = self.flag
         
         # Printing previous and present states
                print("Previous state: ", state_prev)
                print("Present state: ", self.state)
                  
           # Finding the action index
                if act == 1:    
                        self.flag = 1
                        index_prev = self.index
                        self.index = self.dqn_solver_inc.act(self.state)
                        new_cwnd = int(max(obs[6],obs[5]+action_inc[self.index]*(obs[6]*obs[6]/obs[5])))
                        new_ssThresh = int(max(2*obs[6], obs[5]/2))  
                        exec_action = [new_ssThresh, new_cwnd]
                        print("---action taken: ", action_inc[self.index])
                        print("--cwnd before change: ", obs[5])
                        print("--cwnd after change: ",new_cwnd)
                else: 
                        #reward = -5
                        self.flag = 2
                        index_prev = self.index
                        self.index = self.dqn_solver_dec.act(self.state)
                       # new_cwnd = int(max(obs[6],new_cwnd+(new_cwnd/action_dec[index])))
                        new_cwnd = int(max(obs[6],obs[5]+action_dec[self.index]*(obs[6]*obs[6]/obs[5])))
                        new_ssThresh = int(max(2*obs[6], obs[5]/2))  
                        exec_action = [new_ssThresh, new_cwnd]
                        print("---action: ", action_dec[self.index])
                        print("--cwnd (obs[5]): ", obs[5])
                        print("--cwnd (new_cwnd): ",new_cwnd)

                file = open("CWND_nn2_" + str(obs[0]) + ".txt", "a")
                file.write(str(self.line_count))
                file.write(' ')
                file.write(str(new_cwnd))
                file.write("\n")
                file.close()
             # If the line count is even, it means we have a present and next state, so we can do the remember and experience replay. 
                if(self.line_count >= 2):
                        print("Remembering and exprience replay")
                        if flag_prev == 1:
                                self.dqn_solver_inc.remember(state_prev, index_prev, reward_prev, self.state, self.done[0])
                        if flag_prev == 2:
                                self.dqn_solver_dec.remember(state_prev, index_prev, reward_prev, self.state, self.done[0])
                        print("state_prev,index_prev,reward_prev,state_present,done_prev" ,state_prev, index_prev, reward_prev, self.state, self.done[0])
                     # if current state is not last state then do the experience replay   
                        if not(self.done[1]):
                                if self.flag == 1:
                                        self.dqn_solver_inc.experience_replay()
                                if self.flag == 2:
                                        self.dqn_solver_dec.experience_replay()
                
                self.line_count = self.line_count + 1
                
                return exec_action


# initialize variable
get_agent.tcpAgents = {}
get_agent.ob_space = ob_space
get_agent.ac_space = ac_space


obs = env.reset()
print(obs)
print("size of obs space: ",env.observation_space.shape[0])



#print("Observation_space : ",observation_space)
#DQNSolver(Sate_space_size,Action_space_size)

action_inc = [0, 1, 5, 10]
action_dec = [-2, -4]

action_space_inc = len(action_inc)
action_space_dec = len(action_dec)
# Creating a dictionary for maintaining nodes list
nodeList = {}

try:
    while True:
        print("Start iteration: ", currIt)
        print("Step: ", stepIdx)
        #print("---obs: ", observation_space)
        obs = env.reset()
        print("obs: ",obs)

        tcpAgent = get_agent(obs)
              
        while True:
            stepIdx += 1
            # action = tcpAgent.get_action(obs, reward, done, info)
            
            print("Step", stepIdx)
            
            if obs[0] not in nodeList.keys():
                print("Node added")
                node = neuralNet(action_space_inc,action_space_dec)
                nodeList[obs[0]] = node
                print("Dictionary now is ",nodeList)
            
            # Run neuralNet to get an action
            Object = nodeList.get(obs[0])
            exec_action = Object.run(obs)            
            
            obs, reward_1, done, info = env.step(exec_action)
            
            print("---obs, reward_1, done, info: ", obs, reward_1, done, info)
             
            if done:
                stepIdx = 0
                if currIt + 1 < iterationNum:
                    env.reset()
                break
            
        currIt += 1
        if currIt == iterationNum:
            break
        



except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    env.close()
    print("Done")
