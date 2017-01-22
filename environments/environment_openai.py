#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym

import environments.environment

class EnvironmentOpenAI(environments.environment.Environment):

    environment = None

    def __init__(self, environment_id):
        self.environment_id = environment_id
        self.environment = gym.make(self.environment_id)

    def reset(self):
        return self.environment.reset()

    def get_action_space(self):
        action_space = self.environment.action_space
        _class = action_space.__class__
        _class_name = _class.__name__
        if _class_name == "Discrete": return list(range(action_space.n))
        raise Exception("unknown action space: %s" % _class_name)

    def get_maximum_steps(self):
        return self.environment.spec.timestep_limit 

    def do_step(self, action):
        return self.environment.step(action)

    def render(self, stats = []):
        self.environment.render()
