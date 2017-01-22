#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import curses
import random
import signal

import environments.environment

# TODO: add support for assymetric fields
class EnvironmentMinefield(environments.environment.Environment):
   
    class Actions():
        UP = 0
        RIGHT = 1
        BOTTOM = 2
        LEFT = 3

        @classmethod
        def get_actions(cls):
            return (
                cls.UP, 
                cls.RIGHT, 
                cls.BOTTOM, 
                cls.LEFT
            )

    CELL_CHARACTER = 'X'

    CELL_TEMPLATES = dict(
        _ = dict(
            reward = -1,
            empty = True
        ),
        G = dict(
            reward = 100,
            done = True,
            fixed = True,
            position = -1
        ),
        M = dict(
            reward = -100,
            done = True,
            probability = 0.1
        ),
        O = dict(
            reward = 100,
            perishable = True,
            probability = 0.1
        )
    )

    minefield_template = []

    minefield = []

    size = 0

    position = (0, 0)

    done = False

    reward = 0

    screen = None

    def __init__(self, size):
        self.size = size
        
        self._generate_minefield()

    def reset(self):
        self.minefield = list(self.minefield_template)
        self.position = (0, 0)
        self.done = False
        self.reward = 0
        if self.screen: curses.endwin()
        return self.position

    def get_action_space(self):
        return EnvironmentMinefield.Actions.get_actions()

    def get_maximum_steps(self):
        return float('inf')

    def do_step(self, action):
        if action == self.Actions.UP: self.move_up()
        elif action == self.Actions.RIGHT: self.move_right()
        elif action == self.Actions.BOTTOM: self.move_bottom()
        elif action == self.Actions.LEFT: self.move_left()

        x, y = self.position
        cell = self._get_cell(x, y)
        empty_cells = self._get_cell_templates("empty", True)
        empty_cell = random.choice(empty_cells)
        perishable = self._is_cell_attribute(cell, "perishable", True)
        if perishable: self._set_cell(x, y, empty_cell)

        self.reward = self._get_cell_attribute(cell, "reward", 0)
        self.done = self._get_cell_attribute(cell, "done", False)
        observation = self.get_observation()
        return observation

    def render(self, stats = []):
        if not self.screen:
            self.screen = curses.initscr()
            self.screen.border(0)

            def signal_handler(signal, frame):
                curses.endwin()
                sys.exit(0)
            signal.signal(signal.SIGINT, signal_handler)

        for y in range(self.size):
            for x in range(self.size):
                if (x, y) == self.position: cell = self.CELL_CHARACTER
                else: cell = self._get_cell(x, y)
                self.screen.addstr(y + 1, x + 2, cell)

        y = 0
        for name, value in stats:
            self.screen.addstr(y + 1, self.size + 6, "%s: %s" % (name, value))
            y += 1

        self.screen.refresh()

    def get_observation(self):
        x, y = self.position
        state = (
            self._get_cell(x, y, _assert = False),
            self._get_cell(x + 1, y, _assert = False),
            self._get_cell(x - 1, y, _assert = False),
            self._get_cell(x, y + 1, _assert = False),
            self._get_cell(x + 1, y + 1, _assert = False),
            self._get_cell(x - 1, y + 1, _assert = False),
            self._get_cell(x, y - 1, _assert = False),
            self._get_cell(x + 1, y - 1, _assert = False),
            self._get_cell(x - 1, y - 1, _assert = False)
        )
        return (state, self.reward, self.done, None)

    def move_up(self):
        x, y = self.position
        if y - 1 < 0: return
        self.position = (x, y - 1)

    def move_right(self):
        x, y = self.position
        if x + 1 >= self.size: return
        self.position = (x + 1, y)

    def move_bottom(self):
        x, y = self.position
        if y + 1 >= self.size: return
        self.position = (x, y + 1)

    def move_left(self):
        x, y = self.position
        if x - 1 < 0: return
        self.position = (x - 1, y)

    def _get_cell_templates(self, attribute, value):
        cell_templates = []
        for cell, attributes in self.CELL_TEMPLATES.items():
            _value = attributes.get(attribute, None)
            if not _value == value: continue
            cell_templates.append(cell)
        return cell_templates

    def _get_cell_attribute(self, cell, attribute, default = None):
        attributes = self.CELL_TEMPLATES[cell]
        value = attributes.get(attribute, default)
        return value

    def _is_cell_attribute(self, cell, attribute, value):
        _value = self._get_cell_attribute(cell, attribute)
        return _value == value

    def _get_cell(self, x, y, _assert = True):
        try: 
            index = y * self.size + x
            return self.minefield[index]
        except: 
            if _assert: raise

    def _set_cell(self, x, y, cell):
        index = y * self.size + x
        self._set_cell_index(index, cell)

    def _set_cell_index(self, index, cell):
        self.minefield[index] = cell

    def _generate_minefield(self):
        used_cells = []

        self.minefield = [0] * (self.size * self.size)
        empty_cells = self._get_cell_templates("empty", True)
        used_cells += empty_cells
        for x in range(self.size):
            for y in range(self.size):
                cell = random.choice(empty_cells)
                self._set_cell(x, y, cell)

        fixed_cells = self._get_cell_templates("fixed", True)
        used_cells += fixed_cells
        for cell in fixed_cells:
            position = self._get_cell_attribute(cell, "position", 0)
            self._set_cell_index(position, cell)

        remaining_cells = [cell for cell in self.CELL_TEMPLATES.keys() if not cell in used_cells]
        for x in range(self.size):
            for y in range(self.size):
                _cell = self._get_cell(x, y)
                if self._is_cell_attribute(_cell, "empty", False): continue
                for cell in remaining_cells:
                    probability = self._get_cell_attribute(cell, "probability", None)
                    if not probability: continue
                    if random.random() > probability: continue
                    self._set_cell(x, y, cell)
                    break

        self.minefield_template = list(self.minefield)
