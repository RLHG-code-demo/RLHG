from tf_env import tf
from fake_data import FakeData
from HumanEnhanceModule import HumanEnhanceModule
from Agent import Agent
from config import *
import random

fake_data = FakeData(batch_size=TestConfig.batch_size, lstm_step=TestConfig.lstm_step, human_num=TestConfig.human_num, agent_num=TestConfig.agent_num)

size_info = fake_data.get_size()
human_datas, agent_datas = fake_data.get_data()
human_placeholders, agent_placeholders = fake_data.get_placeholder()
human_ph_data_dict = fake_data.get_placeholder_data_dict(human_placeholders, human_datas)
agent_ph_data_dict = fake_data.get_placeholder_data_dict(agent_placeholders, agent_datas)
ph_data_dict = {**human_ph_data_dict, **agent_ph_data_dict}
task_gate_placeholder = tf.placeholder(tf.float32, (1, ), name='task_gate')
ph_data_dict[task_gate_placeholder] = [1.0]

### HumanEnhanceModule ###
feature_list = ['spatial', 'hero', 'monster', 'stat']
spatial_size, hero_size, monster_size, stat_size = [size_info[k] for k in feature_list]
human_enhance_module = HumanEnhanceModule(spatial_size=spatial_size, hero_size=hero_size, monster_size=monster_size, stat_size=stat_size)
human_enhance_module.build(is_pretrain_phase=False)
_, _, human_embedding_list = human_enhance_module.infer(human_feature_list=human_placeholders)

# If there are multiple humans, only one is randomly selected for enhancement.
enhanced_human_index = random.choice(range(Config.human_num))
human_embedding = human_embedding_list[enhanced_human_index]

### Agent ###
feature_list = ['spatial', 'hero', 'monster', 'turret', 'minion', 'stat', 'invisible']
spatial_size, hero_size, monster_size, turret_size, minion_size, stat_size, invisible_size = [size_info[k] for k in feature_list]
agent = Agent(spatial_size=spatial_size, hero_size=hero_size, monster_size=monster_size, turret_size=turret_size, minion_size=minion_size, stat_size=stat_size, invisible_size=invisible_size)
agent.build(is_train=True)
agent_action_list, agent_value_list = agent.infer(agent_feature_list=agent_placeholders, human_embedding=human_embedding, task_gate=task_gate_placeholder)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    result_val = sess.run(agent_action_list, feed_dict=ph_data_dict)
    print([rs[0].shape for rs in result_val])
    result_val = sess.run(agent_value_list, feed_dict=ph_data_dict)
    print([rs.shape for rs in result_val])
