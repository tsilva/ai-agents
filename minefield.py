import agents
import trainers
import environments

# configure the environment where training/testing is going to be done
environment = environments.EnvironmentMinefield(20)

# configure the agent that's going to attempt to 
# maximize the reward in the previous environment
agent = agents.AgentQLearning(
    learning_rate = 1.0,
    future_reward_discount_rate = 1.0,
    exploration_rate = 0.5
)

# train the agent on the environment (returns the parameters 
# for the episode when the agent got the best reward)
_trainer = trainers.Trainer(environment, agent, episode_window = 1000)
best_parameters = _trainer.train()

# load the best parameters into the agent and 
# test him on the same environment (no training is done 
# in the test phase so the parameters won't change)
agent.set_parameters(best_parameters)
_trainer.test()
