#!/usr/bin/python
# -*- coding: utf-8 -*-

class Environment():

    def reset(self):
        raise Exception("not implemented")

    def get_action_space(self):
        raise Exception("not implemented")

    def get_maximum_steps(self):
        raise Exception("not implemented")

    def do_step(self, action):
        raise Exception("not implemented")

    def render(self, stats = []):
        raise Exception("not implemented")
