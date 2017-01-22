#!/usr/bin/python
# -*- coding: utf-8 -*-

import time
import statistics

import utils.queue

# TODO: improve convergence criteria
# TODO: calculate episodes per second
# TODO: calculate steps per second
# TODO: move exit signal handler to trainer
# TODO: load/save agent and environment
class Trainer():

    environment = None
    """ The environment where the 
    agent is going to train """

    agent = None
    """ The agent that will try to maximize the reward 
    in the environment (gives action for each state) """

    training = True
    """ Flag indicating if the trainer is training or testing """

    done = False
    """ Flag indicating if the current episode is finished """

    episode = 0
    """ The number of the current episode """

    episode_reward = 0
    """ The total rewards received for the current episode """

    episode_rewards = None
    """ The last episode rewards """

    step = 0
    """ The number of the last step executed in the current episode """

    step_reward = 0
    """ The reward received by performing the last step """

    best_episode = None
    """ The episode where best reward was received until now """

    best_episode_reward = None
    """ The reward received in the best episode """

    best_parameters = None
    """ The parameters the agent had in the best episode """

    episode_window = None

    def __init__(self, environment, agent, episode_window=100):
        self.environment = environment
        self.agent = agent
        self.episode_window = episode_window
        self.reset()
        
    def reset(self):
        self.training = True
        self.done = False
        self.episode = 0
        self.episode_reward = 0
        self.episode_rewards = utils.queue.Queue(self.episode_window)
        self.step = 0
        self.step_reward = 0
        self.best_episode_reward = 0
        self.best_episode = None
        self.best_parameters = None
        self.agent.reset()

    def train(self):
        self.reset()
        self.training = True
        while not self.is_solved():
            self.start_episode()
            while not self.done: self.do_step()
            self.finish_episode()
        return self.best_parameters

    def test(self):
        self.reset()
        self.training = False
        while True:
            self.start_episode()
            while not self.done: self.do_step()
            self.finish_episode()

    def start_episode(self):
        # increment the episode and log it
        self.episode += 1

        # reset the environment and 
        # retrieve the first state
        self.state = self.environment.reset()

        self.step = 0
        self.step_reward = 0
        self.done = False
        self.episode_reward = 0

    def do_step(self):
        # increment the step and log it
        self.step += 1

        # ask the agent for the next step (taking 
        # into account the resulting state 
        # and reward from the previous action)
        # and execute the step to get to the new state
        actions = self.environment.get_action_space()
        previous_state = self.state
        action = self.agent.do_step(self.state, actions, training=self.is_training())
        self.state, self.step_reward, self.done, info = self.environment.do_step(action)
        if self.is_training(): self.agent.teach(self.state, self.step_reward, actions)

        # render the environments
        if self.is_testing(): 
            stats = self.get_stats()
            self.environment.render(stats=stats)
            time.sleep(0.05)

        # update the episode reward, and in case the total 
        # reward for this episode was the best yet, then 
        # store the agent's parameters (best agent 
        # intelligence until now)
        self.episode_reward += self.step_reward

        if self.is_training():
            values = self.episode_rewards.get_values()
            mean = values and statistics.mean(values) or 0
            print(
                "Training - Episode: %s, Step: %s, Avg. Ep. Reward: %.2f" % (
                    self.episode, self.step, mean
                ),
                end="\r"
            )

    def finish_episode(self):
        self.episode_rewards.append(self.episode_reward)
        if self.episode_reward > self.best_episode_reward:
            self.best_episode = self.episode
            self.best_episode_reward = self.episode_reward
            self.best_parameters = self.agent.get_parameters()

    def is_training(self):
        return self.training

    def is_testing(self):
        return not self.training

    def is_solved(self):
        if self.episode <= self.episode_window: return False
        if self.episode % self.episode_window != 0: return False
        episode_rewards = self.episode_rewards.get_values()
        return self.best_episode_reward >= max(episode_rewards)

    def get_stats(self):
        return (
            ('Agent', self.agent),
            ('Training', self.is_training()),
            ('Testing', self.is_testing()),
            ('Episode', self.episode),
            ('Step', self.step),
            ('Episode Reward', self.episode_reward),
            ('Best Episode', self.best_episode),
            ('Best Episode Reward', self.best_episode_reward)
        )
