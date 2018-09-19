
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
import numpy as np
import theano
import theano.tensor as TT
from lasagne.updates import adam

# normalize() 确保环境的操作在[-1，1]范围内(仅适用于具有连续操作的环境)
env = normalize(CartpoleEnv())
# 用8个隐藏单元的单隐层初始化一个神经网络策略
policy = GaussianMLPPolicy(env.spec, hidden_sizes=(8,))

# 我们将收集每一次迭代的100个轨迹。
N = 100
# 每个轨道最多有100个时间步长。
T = 100
# 迭代次数
n_itr = 100
# 设置问题的折扣因子
discount = 0.99
# 梯度更新的学习速率
learning_rate = 0.01

# 构建计算图形

# 创建用于存储所述观测theano变量
#对于这个例子，我们可以简单地编写 `observations_var = TT.matrix('observations')`。
# 但是，以一种稍微抽象的方式执行它，可以让我们委托环境来处理变量的正确数据类型。
# 例如，对于具有离散观测的环境，我们可能需要使用整数。
observations_var = env.observation_space.new_tensor_variable(
    'observations',
    # 它应该有一个额外的维度，因为我们想要表示一个观察列表。
    extra_dims=1
)
actions_var = env.action_space.new_tensor_variable(
    'actions',
    extra_dims=1
)
returns_var = TT.vector('returns')##

# policy.dist_info_sym 返回一个字典，其值是与操作分布相关的数量的符号表达式。
# 对于高斯策略，它包含标准差的均值和对数。
dist_info_vars = policy.dist_info_sym(observations_var)

# policy.distribution returns a distribution object under rllab.distributions. 
# 它包含许多用于计算与分发相关的数量的实用程序，给定计算出的dist_info_vars. 
# 下面，我们使用dist.log_likelihood_sym 来计算symbolic log-likelihood. 
# 在这个例子中，相应的分布是类rllab.distributions.DiagonalGaussian的一个实例。
dist = policy.distribution

# #请注意，我们否定目标，因为大多数优化器假设最小化问题。
surr = - TT.mean(dist.log_likelihood_sym(actions_var, dist_info_vars) * returns_var)

# 获取可训练参数的列表
params = policy.get_params(trainable=True)
grads = theano.grad(surr, params)

f_train = theano.function(
    inputs=[observations_var, actions_var, returns_var],
    outputs=None,
    updates=adam(grads, params, learning_rate=learning_rate),
    allow_input_downcast=True
)


#收集样本
for _ in range(n_itr):

    paths = []

    for _ in range(N):
        observations = []
        actions = []
        rewards = []

        observation = env.reset()

        for _ in range(T):
            # policy.get_action() 返回一对值。 第二个返回字典，其值包含足够的动作分布统计信息。
            # 它至少应该包含通过调用policy.dist_info()来返回的条目，后者是policy.dist_info_sym()的非符号模拟。
            # 存储这些统计数据是有用的，例如，在形成重要抽样比率时。在我们的例子中，它是不需要的。
            action, _ = policy.get_action(observation)
            # 回想一下，元组的最后一个条目存储有关环境的诊断信息。在我们的情况下，这是不必要的。
            next_observation, reward, terminal, _ = env.step(action)
            observations.append(observation)     #append()列表末尾添加新对象
            actions.append(action)
            rewards.append(reward)
            observation = next_observation
            if terminal:
                # Finish rollout if terminal state reached如果到达终端状态，则完成显示
                break

        # 我们需要计算沿着轨道的每一个时间步骤的经验回报。
        returns = []
        return_so_far = 0
        for t in range(len(rewards) - 1, -1, -1):
            return_so_far = rewards[t] + discount * return_so_far
            returns.append(return_so_far)
        # 返回是向后存储的，所以我们需要恢复它。
        returns = returns[::-1]

        paths.append(dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            returns=np.array(returns)
        ))

#根据经验策略梯度公式，数据向量化
    observations = np.concatenate([p["observations"] for p in paths])
    actions = np.concatenate([p["actions"] for p in paths])
    returns = np.concatenate([p["returns"] for p in paths])

    f_train(observations, actions, returns)
    print('Average Return:', np.mean([sum(p["rewards"]) for p in paths]))
