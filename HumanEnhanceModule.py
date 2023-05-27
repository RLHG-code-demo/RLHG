from tf_env import tf
from utils import _fc_weight_variable, _bias_variable, _conv_weight_variable, maxpooling
import numpy as np
from config import Config, HumanEnhanceConfig

class HumanEnhanceModule:
    def __init__(self, spatial_size, hero_size, monster_size, stat_size):
        self.lstm_time_steps = Config.lstm_step
        self.lstm_unit_size = HumanEnhanceConfig.lstm_unit_size
        self.batch_size = Config.batch_size

        self.spatial_size = spatial_size
        self.hero_size = hero_size
        self.monster_size = monster_size
        self.stat_size = stat_size

        self.human_num = Config.human_num
        self.hero_dim = HumanEnhanceConfig.hero_dim
        self.unit_dim = HumanEnhanceConfig.unit_dim
        self.stat_dim = HumanEnhanceConfig.stat_dim
        self.human_embedding_size = HumanEnhanceConfig.human_embedding_size

        self.value_head = HumanEnhanceConfig.value_head_list

    def build(self, is_pretrain_phase):
        with tf.variable_scope('HumanEnhance'):
            self.conv1_kernel = _conv_weight_variable(shape=[5, 5, self.spatial_size[0], 18], name="spatial_conv1_kernel", trainable=is_pretrain_phase)
            self.conv1_bias = _bias_variable(shape=[18], name="spatial_conv1_bias", trainable=is_pretrain_phase)
            self.conv2_kernel = _conv_weight_variable(shape=[3, 3, 18, 12], name="spatial_conv2_kernel", trainable=is_pretrain_phase)
            self.conv2_bias = _bias_variable(shape=[12], name="spatial_conv2_bias", trainable=is_pretrain_phase)
            spatial_flatten_size = 768

            hero_flatten_size = int(np.prod(self.hero_size))
            self.fc1_hero_weight = _fc_weight_variable(shape=[hero_flatten_size, self.hero_dim], name="fc1_hero_weight", trainable=is_pretrain_phase)
            self.fc1_hero_bias = _bias_variable(shape=[self.hero_dim], name="fc1_hero_bias", trainable=is_pretrain_phase)
            self.fc2_hero_weight = _fc_weight_variable(shape=[self.hero_dim, self.hero_dim // 2], name="fc2_hero_weight", trainable=is_pretrain_phase)
            self.fc2_hero_bias = _bias_variable(shape=[self.hero_dim // 2], name="fc2_hero_bias", trainable=is_pretrain_phase)
            self.fc3_hero_weight = _fc_weight_variable(shape=[self.hero_dim // 2, self.hero_dim // 4], name="fc3_hero_weight", trainable=is_pretrain_phase)
            self.fc3_hero_bias = _bias_variable(shape=[self.hero_dim // 4], name="fc3_hero_bias", trainable=is_pretrain_phase)

            monster_flatten_size = int(np.prod(self.monster_size))
            self.fc1_monster_weight = _fc_weight_variable(shape=[monster_flatten_size, self.unit_dim], name="fc1_monster_weight", trainable=is_pretrain_phase)
            self.fc1_monster_bias = _bias_variable(shape=[self.unit_dim], name="fc1_monster_bias", trainable=is_pretrain_phase)
            self.fc2_monster_weight = _fc_weight_variable(shape=[self.unit_dim, self.unit_dim // 2], name="fc2_monster_weight", trainable=is_pretrain_phase)
            self.fc2_monster_bias = _bias_variable(shape=[self.unit_dim // 2], name="fc2_monster_bias", trainable=is_pretrain_phase)
            self.fc3_monster_weight = _fc_weight_variable(shape=[self.unit_dim // 2, self.unit_dim // 4], name="fc3_monster_weight", trainable=is_pretrain_phase)
            self.fc3_monster_bias = _bias_variable(shape=[self.unit_dim // 4], name="fc3_monster_bias", trainable=is_pretrain_phase)

            stat_flatten_size = int(np.prod(self.stat_size))
            self.fc1_stat_weight = _fc_weight_variable(shape=[stat_flatten_size, self.stat_dim], name="fc1_stat_weight", trainable=is_pretrain_phase)
            self.fc1_stat_bias = _bias_variable(shape=[self.stat_dim], name="fc1_stat_bias", trainable=is_pretrain_phase)
            self.fc2_stat_weight = _fc_weight_variable(shape=[self.stat_dim, self.stat_dim // 4], name="fc2_stat_weight", trainable=is_pretrain_phase)
            self.fc2_stat_bias = _bias_variable(shape=[self.stat_dim // 4], name="fc2_stat_bias", trainable=is_pretrain_phase)

            concat_all_size = spatial_flatten_size + self.hero_dim // 4 + self.unit_dim // 4 + self.stat_dim // 4
            self.fc_concat_weight = _fc_weight_variable(shape=[concat_all_size, self.lstm_unit_size], name="fc_concat_weight", trainable=is_pretrain_phase)
            self.fc_concat_bias = _bias_variable(shape=[self.lstm_unit_size], name="fc_concat_bias", trainable=is_pretrain_phase)

            self.lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.lstm_unit_size, forget_bias=1.0, trainable=is_pretrain_phase)

            self.fc_history_weight = _fc_weight_variable(shape=[self.lstm_unit_size, self.human_embedding_size], name="fc_history_weight", trainable=is_pretrain_phase)
            self.fc_history_bias = _bias_variable(shape=[self.human_embedding_size], name="fc_history_bias", trainable=is_pretrain_phase)

            # value
            self.fc_value_weight = _fc_weight_variable(shape=[self.lstm_unit_size // 4, self.value_head], name="fc_value_weight", trainable=is_pretrain_phase)
            self.fc_value_bias = _bias_variable(shape=[self.value_head], name="fc_value_bias", trainable=is_pretrain_phase)

            # gain
            self.fc1_gain_weight = _fc_weight_variable(shape=[self.lstm_unit_size // 4, self.lstm_unit_size // 2], name="fc1_gain_weight", trainable=not is_pretrain_phase)
            self.fc1_gain_bias = _bias_variable(shape=[self.lstm_unit_size // 2], name="fc1_gain_bias", trainable=not is_pretrain_phase)
            self.fc2_gain_weight = _fc_weight_variable(shape=[self.lstm_unit_size // 2, self.value_head], name="fc2_gain_weight", trainable=not is_pretrain_phase)
            self.fc2_gain_bias = _bias_variable(shape=[self.value_head], name="fc2_gain_bias", trainable=not is_pretrain_phase)

    def infer(self, human_feature_list):
        human_value_list = []
        human_gain_list = []
        human_embedding_list = []
        with tf.variable_scope('HumanEnhance'):
            for player_index in range(self.human_num):
                feature_list = ['spatial', 'hero', 'monster', 'stat']
                spatial, hero, monster, stat = [human_feature_list[player_index][k] for k in feature_list]

                transpose_spatial = tf.transpose(spatial, perm=[0, 2, 3, 1], name='transpose_spatial_%d' % player_index)
                conv1_result = tf.nn.relu((tf.nn.conv2d(transpose_spatial, self.conv1_kernel, strides=[1, 1, 1, 1], padding="SAME") + self.conv1_bias), name="conv1_result_%d" % player_index)
                pool_conv1_result = tf.nn.max_pool(conv1_result, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name="pool_conv1_result_%d" % player_index)
                temp_conv2_result = tf.nn.bias_add(tf.nn.conv2d(pool_conv1_result, self.conv2_kernel, strides=[1, 1, 1, 1], padding="SAME"), self.conv2_bias, name="temp_conv2_result_%d" % player_index)
                conv2_result = tf.transpose(temp_conv2_result, perm=[0, 3, 1, 2], name="conv2_result_%d" % player_index)
                conv2_dim = int(np.prod(conv2_result.get_shape()[1:]))
                flatten_conv2_result = tf.reshape(conv2_result, shape=[-1, conv2_dim], name="flatten_conv2_result_%d" % player_index)

                ### unit feature ###
                # For intuitive display, we first set the number of units of each type to 1.
                # Units of the same type share parameters and will be merged by max pooling.
                fc1_hero_result = tf.nn.relu((tf.matmul(hero, self.fc1_hero_weight) + self.fc1_hero_bias), name="fc1_hero_result_%d" % player_index)
                fc2_hero_result = tf.nn.relu((tf.matmul(fc1_hero_result, self.fc2_hero_weight) + self.fc2_hero_bias), name="fc2_hero_result_%d" % player_index)
                fc3_hero_result = tf.add(tf.matmul(fc2_hero_result, self.fc3_hero_weight), self.fc3_hero_bias, name="fc3_hero_result_%d" % player_index)
                pool_hero_result = maxpooling([fc3_hero_result], 1, self.hero_dim // 4, name="hero_units_%d" % player_index)

                fc1_monster_result = tf.nn.relu((tf.matmul(monster, self.fc1_monster_weight) + self.fc1_monster_bias), name="fc1_monster_result_%d" % player_index)
                fc2_monster_result = tf.nn.relu((tf.matmul(fc1_monster_result, self.fc2_monster_weight) + self.fc2_monster_bias), name="fc2_monster_result_%d" % player_index)
                fc3_monster_result = tf.add(tf.matmul(fc2_monster_result, self.fc3_monster_weight), self.fc3_monster_bias, name="fc3_monster_result_%d" % player_index)
                pool_monster_result = maxpooling([fc3_monster_result], 1, self.unit_dim // 4, name="monster_units_%d" % player_index)

                ### in-game stats feature ###
                fc1_stat_result = tf.nn.relu((tf.matmul(stat, self.fc1_stat_weight) + self.fc1_stat_bias), name="fc1_stat_result_%d" % player_index)
                fc2_stat_result = tf.add(tf.matmul(fc1_stat_result, self.fc2_stat_weight), self.fc2_stat_bias, name="fc2_stat_result_%d" % player_index)

                concat_all = tf.concat([flatten_conv2_result, pool_hero_result, pool_monster_result, fc2_stat_result], axis=1, name="concat_all")
                fc_concat_result = tf.nn.relu((tf.matmul(concat_all, self.fc_concat_weight) + self.fc_concat_bias), name="fc_concat_result_%d" % player_index)

                player_embed_result = tf.concat(fc_concat_result, axis=1, name="embed_concat_result_%d" % player_index)
                reshape_embed_result = tf.reshape(player_embed_result, [-1, self.lstm_time_steps, self.lstm_unit_size], name="reshape_embed_concat_result")

                # lstm
                lstm_initial_state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)
                lstm_outputs, lstm_last_states = tf.nn.dynamic_rnn(self.lstm, reshape_embed_result, initial_state=lstm_initial_state)
                reshape_lstm_outputs_result = tf.reshape(lstm_outputs, [-1, self.lstm_unit_size], name="reshape_lstm_outputs_result")

                human_embedding = tf.nn.relu(tf.matmul(reshape_lstm_outputs_result, self.fc_history_weight) + self.fc_history_bias, name="player%d_fc_history_result" % (player_index))
                human_embedding_list.append(human_embedding)

                # predict value
                with tf.variable_scope("player%d_value" % player_index):
                    fc_value_result = tf.add(tf.matmul(human_embedding, self.fc_value_weight), self.fc_value_bias, name="player%d_fc_value_result" % (player_index))
                    human_value_list.append(fc_value_result)

                # predict gain
                with tf.variable_scope("player%d_gain" % player_index):
                    fc1_gain_result = tf.nn.elu(tf.matmul(tf.stop_gradient(human_embedding), self.fc1_gain_weight) + self.fc1_gain_bias, name="player%d_fc1_gain_result" % (player_index))
                    fc2_gain_result = tf.abs(tf.matmul(fc1_gain_result, self.fc2_gain_weight) + self.fc2_gain_bias, name="player%d_fc2_gain_result" % (player_index))
                    human_gain_list.append(fc2_gain_result)

        return human_value_list, human_gain_list, human_embedding_list