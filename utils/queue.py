#!/usr/bin/python
# -*- coding: utf-8 -*-

class Queue():
    
    values = []

    size = None

    def __init__(self, size):
        self.queue = []
        self.size = size

    def append(self, value):
        if len(self.values) == self.size: self.values = self.values[1:]
        self.values.append(value)

    def is_full(self):
        return len(self.values) == self.size

    def get_values(self):
        return self.values
