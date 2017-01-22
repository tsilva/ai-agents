#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import numpy.random

import agents.agent

# TODO: add support for continous states
# TODO: make different exploration methods configurable
class AgentQLearning(agents.agent.Agent):

    q_map = dict()
    """ This map represents what the agent 
    has learned, each key is a tuple with the 
    state and action, and the value is the 
    reward for that state-action combo """

    learning_rate = 0.9
    """ Alpha represents how much the a q value
    should be updated in each iteration """

    exploration_rate = 0.25
    """ The probability that a random action will
    be taken during the training phase, as opposed
    to the optimal action at that point (increases
    probability of finding better solutions but
    may decrease convergence speed) """
    
    future_reward_discount_rate = 0.9
    """ Gamma represents how much future rewards
    should be taken into account when calculating
    the q value for a state-action combo """

    state = None
    """ The last state the agent observed """

    action = None
    """ The last action the agent took after
    observing his state attribute """

    step = 0
    """ The number of steps that the agent has taken 
    before he was asked to reset his state """

    def __init__(
        self, 
        learning_rate = 0.9, 
        exploration_rate = 0.25, 
        future_reward_discount_rate = 0.9
    ):
        self.q_map = dict()
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.future_reward_discount_rate = future_reward_discount_rate
        self.state = None
        self.action = None
        self.step = 0

    def reset(self):
        self.state = None
        self.action = None
        self.step = 0

    def do_step(self, state, actions, training = True):
        self.step += 1
        self.state = state
        self.action = self.get_best_action(self.state, actions, training = training)
        return self.action

    def teach(self, new_state, reward, actions):
        best_next_action = self.get_best_action(new_state, actions, training = False)
        best_next_q = self.get_q(new_state, best_next_action)
        q = self.get_q(self.state, self.action)
        q += self.learning_rate * (reward + self.future_reward_discount_rate * best_next_q - q)
        self.q_map[(self.state, self.action)] = q

    def get_q(self, state, action):
        return self.q_map.get((state, action), 0)

    def get_best_action(self, state, actions, training = True):
        explore = training and random.random() < self.exploration_rate
        if explore: return random.choice(actions)
        return self.get_best_action_greedy(state, actions)

    def get_best_action_greedy(self, state, actions):
        _actions = []
        best_q = float('-inf')
        for action in actions:
            q = self.get_q(state, action)
            if q == best_q: _actions.append(action)
            elif q > best_q: _actions = [action]; best_q = q
        action = random.choice(_actions)
        return action

    def get_best_action_weighted_probability(self, state, actions):
        # add the minimum to every value to 
        # ensure that all values are positive
        q_values = [self.get_q(state, action) for action in actions]
        minimum = min(q_values)
        probabilities = [value + abs(minimum) for value in q_values]

        # divide all values by the total to 
        # ensure they are all between zero and one
        # (while keeping their relative weights 
        # and all sum 1.0)
        total = sum(probabilities)
        probabilities = [value / float(total) if total != 0 else 0 for value in probabilities]

        # in the edge case where all values are zero, choose a 
        # random action to be one (100% probability of choice)
        if(sum(probabilities)) == 0:
            index = random.randint(0, len(probabilities) - 1)
            probabilities[index] = 1

        # choose a random action taking into 
        # account that the ones with a higher 
        # q value should be picked more often
        action = numpy.random.choice(actions, p = probabilities)
        return action

    def get_parameters(self):
        return self.q_map

    def set_parameters(self, parameters):
        self.q_map = parameters

    def print_parameters(self):
        for key in self.q_map.keys():
            state, action = key
            if action == 0: action = "up"
            elif action == 1: action = "right"
            elif action == 2: action = "bottom"
            elif action == 3: action = "left"
            q = self.q_map[key]
            print("(%s, %s) = %s" % (state, action, q))
