from tf_env import tf
import numpy as np

class FakeData:
    def __init__(self, batch_size, lstm_step, human_num, agent_num):
        self.batch_size = batch_size * lstm_step
        self.human_num = human_num
        self.agent_num = agent_num
        self.size_dict = {
            'hero': [100], # Unit feature -- Heros
            'monster': [50],  # Unit feature -- Monsters
            'turret': [50],  # Unit feature -- Turrets
            'minion': [50],  # Unit feature -- Minions
            'stat': [50], # In-game stats feature
            'spatial': [7, 17, 17], # Spatial feature
            'invisible': [50], # Invisible opponent information

            'task_return': [5],  # Task return -- real r + gamma * V(S') for each value head (task reward)
            'human_return': [4],  # Human goal return -- real r + gamma * V(S') for each value head (human goal reward)
        }
        self.human_mask_feature = ['minion', 'turret', 'invisible']

        # Action space: What(14), Move X(9), Move Y(9), Skill X(9), Skill Y(9), Who(5)
        for index, action_size in enumerate([14, 9, 9, 9, 9, 5]):
            # old policy -- real softmax prob
            self.size_dict['old_policy_%d' % index] = [action_size]
            # action label - real selected action (one-hot)
            self.size_dict['action_label_%d' % index] = [action_size]

    def get_size(self):
        return self.size_dict

    def build_player_data(self, is_human):
        data = {}
        for key, size in self.size_dict.items():
            if is_human and key in self.human_mask_feature:
                continue
            if 'action_label' in key:
                data[key] = np.zeros(([self.batch_size] + size))
                rand_idx = np.random.randint(0, size, self.batch_size)
                for row, col in enumerate(rand_idx):
                    data[key][row, col] = 1
            elif 'old_policy' in key:
                x = np.random.rand(*([self.batch_size] + size))
                x = x - np.max(x, axis=1)[:, np.newaxis]
                data[key] = np.exp(x) / np.sum(np.exp(x), axis=1)[:, np.newaxis]
            else:
                data[key] = np.random.rand(*([self.batch_size] + size))
        return data

    def get_data(self):
        human_datas = []
        agent_datas = []
        for i in range(self.human_num):
            human_datas.append(self.build_player_data(is_human=True))
        for i in range(self.agent_num):
            agent_datas.append(self.build_player_data(is_human=False))
        return human_datas, agent_datas

    def build_player_placeholder(self, player_name):
        placeholder = {}
        for key, size in self.size_dict.items():
            if 'human' in player_name and key in self.human_mask_feature:
                continue
            placeholder[key] = tf.placeholder(tf.float32, [self.batch_size] + size, name='%s_%s' % (player_name, key))
        return placeholder

    def get_placeholder(self):
        human_placeholders = []
        agent_placeholders = []
        for i in range(self.human_num):
            human_placeholders.append(self.build_player_placeholder(player_name='human_%d' % i))
        for i in range(self.agent_num):
            agent_placeholders.append(self.build_player_placeholder(player_name='agent_%d' % i))
        return human_placeholders, agent_placeholders

    def get_placeholder_data_dict(self, placeholders, datas):
        ph_data_dict = {}
        assert len(placeholders) == len(datas)
        for placeholder, data in zip(placeholders, datas):
            assert len(placeholder) == len(data)
            for key, size in placeholder.items():
                ph_data_dict[placeholder[key]] = data[key]
        return ph_data_dict