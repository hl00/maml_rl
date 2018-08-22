from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

stub(globals())

env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # 神经网络策略应该有两个隐层，每层都有32个隐藏单元。
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=1000,
    discount=0.99,
    step_size=0.01,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,取消注释两行(此参数和下面的绘图参数)，以启用绘图=True，
)

run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling平行取样工人人数
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration只保留最后一次迭代的快照参数
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed指定实验的种子。如果没有提供，则使用随机种子
    # will be used
    seed=1,
    # plot=True,
)
