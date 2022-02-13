import tiles3 as tc
import numpy as np

class Tilecoder:
    def __init__(self, min_position, max_position, min_velocity, max_velocity,
     iht_size = 4096, num_tilings = 8, num_tiles = 4):
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.min_position = min_position
        self.max_position = max_position

    def get_tiles(self, state):
        position, velocity = state[0], state[1]
        position_scaled = (position - self.min_position)/(self.max_position - self.min_position) * self.num_tiles
        velocity_scaled = (velocity - self.min_velocity)/(self.max_velocity - self.min_velocity) * self.num_tiles
        tiles = tc.tiles(self.iht, self.num_tilings, [position_scaled, velocity_scaled])
        return np.array(tiles)

class ReinforceAgent:
    def agent_init(self, agent_info = {}, adam_info = {}):
        min_position = agent_info.get('min_position')
        max_position = agent_info.get('max_position')
        min_velocity = agent_info.get('min_velocity')
        max_velocity = agent_info.get('max_velocity')
        self.iht_size = agent_info.get('iht_size')
        self.num_tilings = agent_info.get('num_tilings')
        self.num_tiles = agent_info.get('num_tiles')
        self.step_size = agent_info.get('step_size')
        self.discount_factor = agent_info.get('discount_factor')
        self.num_actions = agent_info.get('num_actions')

        self.theta = np.zeros((self.num_actions * self.iht_size, ))
        # For Fixing Left and Right action initially
        #self.theta[0:self.iht_size] = 1
        #self.theta[self.iht_size*2:] = 1

        self.tilecoder = Tilecoder(min_position, max_position, min_velocity, max_velocity,
        iht_size = self.iht_size, num_tilings = self.num_tilings, num_tiles = self.num_tiles)

    def agent_step(self, state):
        action = self.policy(state)
        return action
    
    def agent_end(reward):    
        pass

    def get_state_action_feature(self, state, action):
        """
        Take state and action as input, encode it through tile coding and stack it to produce feature
        action = [0,1,2]
        """
        active_tile = self.tilecoder.get_tiles(state)
        active_feature_per_action = np.zeros((self.iht_size, ))
        active_feature_per_action[active_tile] = 1
        active_state_action_feature = np.zeros((self.num_actions * self.iht_size, ))
        active_state_action_feature[self.iht_size * action:self.iht_size * (action+1)] = active_feature_per_action
        return active_state_action_feature

    def get_action_value(self, active_state_action_feature):
        action_value = np.dot(self.theta, active_state_action_feature)
        return action_value

    def softmax(self, state):
        preferences = []
        for a in range(self.num_actions):
            active_state_action_feature = self.get_state_action_feature(state, a)
            action_value = self.get_action_value(active_state_action_feature)
            preferences.append(action_value)
        preferences = np.array(preferences)
        max_preferences = np.max(preferences)
        preferences = preferences - max_preferences
        exp_preferences = np.exp(preferences)
        sum_exp_preferences  = np.sum(exp_preferences)
        action_probs = exp_preferences / sum_exp_preferences
        return action_probs

    def policy(self, state):
        action_probs = self.softmax(state)
        #print(action_probs)
        action = np.random.choice(self.num_actions, p=action_probs)
        return action

    def get_gradient(self, state, action):
        active_state_action_feature = self.get_state_action_feature(state, action)
        action_probs = self.softmax(state)

        feature_set = []
        for a in range(self.num_actions):
            feature = self.get_state_action_feature(state, a)
            feature_set.append(feature)
        feature_set = np.array(feature_set)
        expected_feature_value = np.dot(action_probs, feature_set)
        
        
        gradient = active_state_action_feature - expected_feature_value
        return gradient

    def update_weight(self, action, state, g, t):
        gradient = self.get_gradient(state, action)
        self.theta = self.theta + self.step_size * (self.discount_factor ** t) * g * gradient


class ReinforceBaseLineAgent:
    def agent_init(self, agent_info = {}, adam_info = {}):
        min_position = agent_info.get('min_position')
        max_position = agent_info.get('max_position')
        min_velocity = agent_info.get('min_velocity')
        max_velocity = agent_info.get('max_velocity')
        self.iht_size = agent_info.get('iht_size')
        self.num_tilings = agent_info.get('num_tilings')
        self.num_tiles = agent_info.get('num_tiles')
        self.step_size_theta = agent_info.get('step_size_theta')
        self.step_size_w = agent_info.get('step_size_w')
        self.discount_factor = agent_info.get('discount_factor')
        self.num_actions = agent_info.get('num_actions')

        self.theta = np.zeros((self.num_actions * self.iht_size, ))
        self.theta[0:self.iht_size] = 1
        self.theta[self.iht_size*2:] = 1

        self.w = np.zeros((self.iht_size, ))

        self.tilecoder = Tilecoder(min_position, max_position, min_velocity, max_velocity,
        iht_size = self.iht_size, num_tilings = self.num_tilings, num_tiles = self.num_tiles)

    def agent_step(self, state):
        action = self.policy(state)
        return action
    
    def agent_end(reward):    
        pass

    def get_state_feature(self, state):
        active_tile = self.tilecoder.get_tiles(state)
        active_state_feature = np.zeros((self.iht_size, ))
        active_state_feature[active_tile] = 1
        return active_state_feature

    def get_state_value(self, active_state_feature):
        state_value = np.dot(self.w, active_state_feature)
        return state_value

    def get_state_action_feature(self, state, action):
        """
        Take state and action as input, encode it through tile coding and stack it to produce feature
        action = [0,1,2]
        """
        active_tile = self.tilecoder.get_tiles(state)
        active_feature_per_action = np.zeros((self.iht_size, ))
        active_feature_per_action[active_tile] = 1
        active_state_action_feature = np.zeros((self.num_actions * self.iht_size, ))
        active_state_action_feature[self.iht_size * action:self.iht_size * (action+1)] = active_feature_per_action
        return active_state_action_feature

    def get_action_value(self, active_state_action_feature):
        action_value = np.dot(self.theta, active_state_action_feature)
        return action_value

    def softmax(self, state):
        preferences = []
        for a in range(self.num_actions):
            active_state_action_feature = self.get_state_action_feature(state, a)
            action_value = self.get_action_value(active_state_action_feature)
            preferences.append(action_value)
        preferences = np.array(preferences)
        max_preferences = np.max(preferences)
        preferences = preferences - max_preferences
        exp_preferences = np.exp(preferences)
        sum_exp_preferences  = np.sum(exp_preferences)
        action_probs = exp_preferences / sum_exp_preferences
        return action_probs

    def policy(self, state):
        action_probs = self.softmax(state)
        #print(action_probs)
        action = np.random.choice(self.num_actions, p=action_probs)
        return action

    def get_gradient(self, state, action):
        active_state_action_feature = self.get_state_action_feature(state, action)
        action_probs = self.softmax(state)

        feature_set = []
        for a in range(self.num_actions):
            feature = self.get_state_action_feature(state, a)
            feature_set.append(feature)
        feature_set = np.array(feature_set)
        expected_feature_value = np.dot(action_probs, feature_set)
        
        
        gradient = active_state_action_feature - expected_feature_value
        return gradient

    def update_theta_weight(self, action, state, delta, t):
        gradient = self.get_gradient(state, action)
        self.theta = self.theta + self.step_size_theta * (self.discount_factor ** t) * delta * gradient

    def update_w_weight(self, state, delta):
        self.w = self.w + self.step_size_w * delta * self.get_state_feature(state)