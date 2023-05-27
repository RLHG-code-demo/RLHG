from tf_env import tf
from fake_data import FakeData
from HumanEnhanceModule import HumanEnhanceModule
from config import *

fake_data = FakeData(batch_size=Config.batch_size, lstm_step=Config.lstm_step, human_num=Config.human_num, agent_num=Config.agent_num)

size_info = fake_data.get_size()
human_datas, _ = fake_data.get_data()
human_placeholders, _ = fake_data.get_placeholder()
ph_data_dict = fake_data.get_placeholder_data_dict(human_placeholders, human_datas)

### HumanEnhanceModule ###
feature_list = ['spatial', 'hero', 'monster', 'stat']
spatial_size, hero_size, monster_size, stat_size = [size_info[k] for k in feature_list]
human_enhance_module = HumanEnhanceModule(spatial_size=spatial_size, hero_size=hero_size, monster_size=monster_size, stat_size=stat_size)
human_enhance_module.build(is_pretrain_phase=True)
human_value_list, _, _ = human_enhance_module.infer(human_feature_list=human_placeholders)

losses = []
### forward ###
for index, human_primitive_value in enumerate(human_value_list):
    player_loss = tf.constant(0.0, dtype=tf.float32)

    human_return = human_placeholders[index]['human_return']
    value_loss = 0.5 * tf.reduce_mean(tf.square(human_return - human_primitive_value), axis=0)
    player_loss += tf.reduce_sum(value_loss)

    losses.append(player_loss)
total_loss = tf.reduce_sum(losses)

### backpropagation ###
params = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(HumanEnhanceConfig.init_learning_rate, beta1=0.9, beta2=0.999, epsilon=0.00001)
grads = tf.gradients(total_loss, params)
train_op = optimizer.apply_gradients(zip(grads, params))

init = tf.global_variables_initializer()
global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(HumanEnhanceConfig.epoch):
        result = sess.run([train_op], feed_dict=ph_data_dict)
        visual_loss = sess.run(total_loss, feed_dict=ph_data_dict)
        print('epoch %d: loss=%f' % (epoch, visual_loss))
        print(sess.run(tf.reduce_sum(human_enhance_module.fc_value_weight)))
        print(sess.run(tf.reduce_sum(human_enhance_module.fc1_gain_weight)))