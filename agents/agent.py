#!/usr/bin/python
# -*- coding: utf-8 -*-

class Agent():

    def reset(self):
        raise Exception("not implemented")

    def do_step(self, state, actions, training = True):
        raise Exception("not implemented")

    def teach(self, state, reward, actions):
        raise Exception("not implemented")

    def get_parameters(self):
        raise Exception("not implemented")

    def set_parameters(self, parameters):
        raise Exception("not implemented")

    def __str__(self):
        return self.__class__.__name__
