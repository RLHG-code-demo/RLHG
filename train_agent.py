from tf_env import tf
from fake_data import FakeData
from HumanEnhanceModule import HumanEnhanceModule
from Agent import Agent
from config import *
import random

losses = []
### Task Selection ###
task_gate_placeholder = tf.placeholder(tf.float32, (1, ), name='task_gate')
# Randomly select human-agent or agent-only mode.
if random.random() <= AgentConfig.beta:
    print('human-agent mode')
    task_gate = 1.0
    # If there are multiple humans, only one is randomly selected for enhancement.
    enhanced_human_index = random.choice(range(Config.human_num))
    fake_data = FakeData(batch_size=Config.batch_size, lstm_step=Config.lstm_step, human_num=Config.human_num, agent_num=Config.agent_num)

    size_info = fake_data.get_size()
    human_datas, agent_datas = fake_data.get_data()
    human_placeholders, agent_placeholders = fake_data.get_placeholder()
    human_ph_data_dict = fake_data.get_placeholder_data_dict(human_placeholders, human_datas)
    agent_ph_data_dict = fake_data.get_placeholder_data_dict(agent_placeholders, agent_datas)
    ph_data_dict = {**human_ph_data_dict, **agent_ph_data_dict}
    ph_data_dict[task_gate_placeholder] = [task_gate]

    ### HumanEnhanceModule ###
    feature_list = ['spatial', 'hero', 'monster', 'stat']
    spatial_size, hero_size, monster_size, stat_size = [size_info[k] for k in feature_list]
    human_enhance_module = HumanEnhanceModule(spatial_size=spatial_size, hero_size=hero_size, monster_size=monster_size, stat_size=stat_size)
    human_enhance_module.build(is_pretrain_phase=False)
    human_value_list, human_gain_list, human_embedding_list = human_enhance_module.infer(human_feature_list=human_placeholders)

    # Calculate the human gain and the human-enhanced advantage
    human_return = agent_placeholders[enhanced_human_index]['human_return']
    human_primitive_value = human_value_list[enhanced_human_index]
    human_real_gain = human_return - human_primitive_value
    human_gain = human_gain_list[enhanced_human_index]
    human_embedding = human_embedding_list[enhanced_human_index]
    human_advantage = tf.reduce_sum(human_real_gain - tf.stop_gradient(human_gain), axis=-1)
    indicator = tf.where(human_real_gain > 0, tf.ones_like(human_real_gain), tf.zeros_like(human_real_gain))

    # Calculate the human gain loss
    human_loss = 0.5 * tf.reduce_mean(indicator * tf.square(human_real_gain - human_gain), axis=0)
    losses.append(tf.reduce_sum(human_loss))
else:
    print('agent-only mode')
    task_gate = 0.0
    fake_data = FakeData(batch_size=Config.batch_size, lstm_step=Config.lstm_step, human_num=0, agent_num=Config.max_player_num)

    size_info = fake_data.get_size()
    _, agent_datas = fake_data.get_data()
    _, agent_placeholders = fake_data.get_placeholder()
    ph_data_dict = fake_data.get_placeholder_data_dict(agent_placeholders, agent_datas)
    ph_data_dict[task_gate_placeholder] = [task_gate]

    human_embedding = tf.zeros([Config.batch_size * Config.lstm_step, AgentConfig.human_embedding_size], dtype=tf.float32)
    human_advantage = 0

### Agent ###
feature_list = ['spatial', 'hero', 'monster', 'turret', 'minion', 'stat', 'invisible']
spatial_size, hero_size, monster_size, turret_size, minion_size, stat_size, invisible_size = [size_info[k] for k in feature_list]
agent = Agent(spatial_size=spatial_size, hero_size=hero_size, monster_size=monster_size, turret_size=turret_size, minion_size=minion_size, stat_size=stat_size, invisible_size=invisible_size)
agent.build(is_train=True)
agent_action_list, agent_value_list = agent.infer(agent_feature_list=agent_placeholders, human_embedding=human_embedding, task_gate=task_gate_placeholder)

### forward ###
for index, (agent_policy, agent_value) in enumerate(zip(agent_action_list, agent_value_list)):
    player_loss = tf.constant(0.0, dtype=tf.float32)

    task_return = agent_placeholders[index]['task_return']
    value_loss = 0.5 * tf.reduce_mean(tf.square(task_return - agent_value), axis=0)
    player_loss += tf.reduce_sum(value_loss)

    ori_advantage = tf.reduce_sum(task_return - tf.stop_gradient(agent_value), axis=-1)
    advantage = ori_advantage + AgentConfig.alpha * human_advantage
    for action_index, policy in enumerate(agent_policy):
        old_policy, action_label = agent_placeholders[index]['old_policy_%d' % action_index], agent_placeholders[index]['action_label_%d' % action_index]

        # print(old_policy, action_label, policy)
        policy_p = tf.reduce_sum(action_label * policy, axis=1)
        policy_log_p = tf.log(policy_p)
        old_policy_p = tf.reduce_sum(action_label * old_policy, axis=1)
        old_policy_log_p = tf.log(old_policy_p)
        ratio = tf.exp(policy_log_p - old_policy_log_p)
        surr1 = tf.clip_by_value(ratio, 0.0, AgentConfig.dual_clip_param) * advantage
        surr2 = tf.clip_by_value(ratio, 1.0 - AgentConfig.clip_param, 1.0 + AgentConfig.clip_param) * advantage
        dual_ppo_loss = - tf.reduce_mean(tf.minimum(surr1, surr2))
        player_loss += dual_ppo_loss

        entropy_loss = -tf.reduce_mean(policy_p * policy_log_p)
        player_loss += entropy_loss
    losses.append(player_loss)
total_loss = tf.reduce_sum(losses)

### backpropagation ###
params = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(AgentConfig.init_learning_rate, beta1=0.9, beta2=0.999, epsilon=0.00001)
grads = tf.gradients(total_loss, params)
train_op = optimizer.apply_gradients(zip(grads, params))

init = tf.global_variables_initializer()
global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(AgentConfig.epoch):
        result = sess.run([train_op], feed_dict=ph_data_dict)
        visual_loss = sess.run(total_loss, feed_dict=ph_data_dict)
        print('epoch %d: loss=%f' % (epoch, visual_loss))
