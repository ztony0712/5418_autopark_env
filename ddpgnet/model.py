from keras.models import Model
from keras.layers import Input, Dense, Concatenate
import numpy as np

RNN_SIZE = 128  # 隐藏层大小
GOAL_REPR = 12  # 目标表示的维度
ACTION_SIZE = 2  # 动作空间大小（例如转向和加速）

def build_ddpg_model():
    # 输入层
    input_state = Input(shape=(6,))  # 状态维度：x, y, vx, vy, cos_h, sin_h
    input_goal = Input(shape=(6,))  # 目标状态：目标位置和朝向

    # Actor 网络
    h1_actor = Dense(RNN_SIZE, activation='relu')(input_state)  # 隐藏层
    h2_actor = Dense(RNN_SIZE, activation='relu')(h1_actor)  # 隐藏层
    action_output = Dense(ACTION_SIZE, activation='tanh')(h2_actor)  # 输出动作

    # Critic 网络
    h1_critic = Dense(RNN_SIZE, activation='relu')(input_state)  # 状态输入的隐藏层
    h2_critic = Dense(RNN_SIZE, activation='relu')(h1_critic)  # 状态隐藏层
    goal_layer = Dense(GOAL_REPR, activation='relu')(input_goal)  # 目标输入的隐藏层

    hidden_input = Concatenate(axis=1)([h2_critic, goal_layer])  # 合并状态和目标表示
    h3_critic = Dense(RNN_SIZE, activation='relu')(hidden_input)  # 合并后的隐藏层
    value_output = Dense(1, activation=None)(h3_critic)  # 输出 Q 值

    # 创建模型
    actor_model = Model(inputs=input_state, outputs=action_output, name='Actor_Model')
    critic_model = Model(inputs=[input_state, input_goal], outputs=value_output, name='Critic_Model')

    return actor_model, critic_model

def test_model(actor_model, critic_model, state_inputs, goal_inputs):
    """
    测试模型，接受多组状态和目标输入，返回相应的动作和 Q 值。

    :param actor_model: Actor 模型
    :param critic_model: Critic 模型
    :param state_inputs: 状态输入数组，形状为 (num_samples, 6)
    :param goal_inputs: 目标输入数组，形状为 (num_samples, 6)
    :return: 动作输出和 Q 值
    """
    actions = actor_model.predict(state_inputs)  # 获取动作输出
    q_values = critic_model.predict([state_inputs, goal_inputs])  # 获取 Q 值

    return actions, q_values

# 示例：构建模型并进行测试
if __name__ == "__main__":
    actor_model, critic_model = build_ddpg_model()

    # 模拟输入
    num_samples = 5  # 测试样本数量
    state_inputs = np.random.rand(num_samples, 6)  # 随机生成状态输入
    goal_inputs = np.random.rand(num_samples, 6)   # 随机生成目标输入

    actions, q_values = test_model(actor_model, critic_model, state_inputs, goal_inputs)

    print("Actions:\n", actions)
    print("Q Values:\n", q_values)
