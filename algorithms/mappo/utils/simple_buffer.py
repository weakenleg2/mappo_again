import torch

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.actions_buffer = []
        self.obs_buffer = []
        self.one_hot_list_buffer = []
        self.rewards_buffer = []
        self.current_size = 0

    def insert(self, action, obs, one_hot_list, reward):
        if self.current_size < self.buffer_size:
            self.actions_buffer.append(action)
            self.obs_buffer.append(obs)
            self.one_hot_list_buffer.append(one_hot_list)
            self.rewards_buffer.append(reward)
            self.current_size += 1
        else:
            # Handle buffer overflow, for example, by removing the oldest data
            self.actions_buffer.pop(0)
            self.obs_buffer.pop(0)
            self.one_hot_list_buffer.pop(0)
            self.rewards_buffer.pop(0)

            self.actions_buffer.append(action)
            self.obs_buffer.append(obs)
            self.one_hot_list_buffer.append(one_hot_list)
            self.rewards_buffer.append(reward)

    def sample(self, batch_size):
        # Implement a method to sample a batch of data from the buffer
        pass

    def clear(self):
        self.actions_buffer.clear()
        self.obs_buffer.clear()
        self.one_hot_list_buffer.clear()
        self.rewards_buffer.clear()
        self.current_size = 0

# Usage
# replay_buffer = ReplayBuffer(buffer_size=10000)

# # Example of inserting data into the buffer
# for step in range(self.episode_length):
#     # ... (your existing code for generating actions, obs, one_hot_list, rewards)
#     replay_buffer.insert(actions, obs, one_hot_list, rewards)

# # Later, you can sample data from the replay buffer for training or analysis
