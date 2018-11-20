
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.algos.vpg import VPG
from sandbox.rocky.tf.policies.minimal_gauss_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from maml_examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

import tensorflow as tf

env = normalize(PointEnvRandGoalOracle())

env = TfEnv(env)
policy = GaussianMLPPolicy(
    name='policy',
    env_spec=env.spec,
    hidden_nonlinearity=tf.nn.relu,
    hidden_sizes=(100, 100)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=1000,
    max_path_length=100,
    n_itr=100,
    discount=0.99,
    step_size=0.01,
    plot=True,
)
#import pdb; pdb.set_trace()
run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=4,
    # 只保留上次迭代的快照参数。
    snapshot_mode="last",
    # 指定实验的种子。如果没有提供，则将使用随机种子。
    seed=1,
    exp_prefix='trpo_maml_point100',
    exp_name='oracleenv2',
    plot=True,
)
