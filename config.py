class Config:
    human_num = 1
    agent_num = 4
    max_player_num = 5
    task_value_head = 5
    human_value_head = 4
    batch_size = 2
    lstm_step = 16

class TestConfig:
    human_num = 1
    agent_num = 4
    batch_size = 2
    lstm_step = 16

class HumanEnhanceConfig:
    hero_dim = 256
    unit_dim = 128
    stat_dim = 128
    lstm_unit_size = 4096
    human_embedding_size = lstm_unit_size // 4

    # human goal reward
    value_head_list = Config.human_value_head

    init_learning_rate = 0.0001
    epoch = 2000000

class AgentConfig:
    hero_dim = 512
    unit_dim = 256
    stat_dim = 256
    invisible_dim = 256
    action_query_dim = 128
    action_key_dim = 128
    lstm_unit_size = 4096
    human_embedding_size = HumanEnhanceConfig.human_embedding_size

    # What, How (Move X), How (Move Y), Skill (Move X), Skill (Move Y), Who (Target Unit)
    action_size_list = [14, 9, 9, 9, 9, 5]

    # task reward
    value_head_list = Config.task_value_head

    init_learning_rate = 0.0001
    epoch = 2000000

    clip_param = 0.2
    dual_clip_param = 3

    alpha = 2
    beta = 0.5
