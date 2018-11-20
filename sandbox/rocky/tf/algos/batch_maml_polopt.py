import matplotlib
matplotlib.use('Pdf')

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import rllab.misc.logger as logger
import rllab.plotter as plotter
import tensorflow as tf
import time

from rllab.algos.base import RLAlgorithm
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.samplers.batch_sampler import BatchSampler
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler
from sandbox.rocky.tf.spaces import Discrete
from rllab.sampler.stateful_pool import singleton_pool

class BatchMAMLPolopt(RLAlgorithm):
    """
    Base class for batch sampling-based policy optimization methods, with maml.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    基于批量采样的策略优化方法的基类，带有maml。
     这包括各种政策梯度方法，如vpg，npg，ppo，trpo等。
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            # 请注意，grad upate的轨迹数 = batch_size
            #默认值为10个轨迹，长度为500，用于梯度更新

            batch_size=100,
            max_path_length=500,
            meta_batch_size = 100,
            num_grad_updates=1,
            discount=0.99,
            gae_lambda=1,
            plot=True,
            pause_for_plot=False,
            center_adv=True,
            positive_adv=False,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            use_maml=True,
            load_policy=None,
            **kwargs
    ):
        """
        :param env: Environment
        :param policy: Policy
        :type policy: Policy
        :param baseline: Baseline
        :param scope: Scope for identifying the algorithm. Must be specified if running multiple algorithms
        simultaneously, each using different environments and policies
        :param n_itr: Number of iterations.
        :param start_itr: Starting iteration.
        :param batch_size: Number of samples per iteration.  #每次迭代的样本数
        :param max_path_length: 单个卷展栏rollout的最大长度
        :param meta_batch_size: 每次元更新采样的任务数
        :param num_grad_updates: Number of fast gradient updates快速渐变更新的数量
        :param discount: Discount.
        :param gae_lambda: Lambda used for generalized advantage estimation.Lambda用于广义优势估计。
        :param plot: Plot evaluation run after each iteration.
        :param pause_for_plot: Whether to pause before contiuing when plotting.
        :param center_adv: Whether to rescale the advantages so that they have mean 0 and standard deviation 1.是否重新调整优势，使它们具有均值0和标准差1。
        :param positive_adv: Whether to shift the advantages so that they are always positive. When used in
        conjunction with center_adv the advantages will be standardized before shifting.
        :param store_paths: Whether to save all paths data to the snapshot.
        :return:
        """
        self.env = env
        self.policy = policy
        self.load_policy=load_policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        # batch_size是一次快速梯度更新的轨迹数
        # self.batch_size 是要收集的总转换数transitions
        self.batch_size = batch_size * max_path_length * meta_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.meta_batch_size = meta_batch_size  # 任务数量
        self.num_grad_updates = num_grad_updates # 训练期间的梯度步数gradient steps

        if sampler_cls is None:
            if singleton_pool.n_parallel > 1:
                sampler_cls = BatchSampler
            else:
                sampler_cls = VectorizedSampler
        if sampler_args is None:
            sampler_args = dict()
        sampler_args['n_envs'] = self.meta_batch_size
        self.sampler = sampler_cls(self, **sampler_args)

    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)

    def shutdown_worker(self):
        self.sampler.shutdown_worker()

    def obtain_samples(self, itr, reset_args=None, log_prefix=''):
        # 这使用self.policy获取样本，并调用policy.get_actions（obses）
        # return_dict specifies how the samples should be returned (dict separates samples
        # by task)return_dict指定应如何返回样本（dict按任务分隔样本）
        paths = self.sampler.obtain_samples(itr, reset_args, return_dict=True, log_prefix=log_prefix)
        assert type(paths) == dict
        return paths

    def process_samples(self, itr, paths, prefix='', log=True):
        return self.sampler.process_samples(itr, paths, prefix=prefix, log=log)

    def train(self):
        # TODO - make this a util
        flatten_list = lambda l: [item for sublist in l for item in sublist]

        with tf.Session() as sess:
            # Code for loading a previous policy. Somewhat hacky because needs to be in sess.用于加载先前策略的代码。
            if self.load_policy is not None:
                import joblib
                self.policy = joblib.load(self.load_policy)['policy']
            self.init_opt()
            # initialize uninitialized vars  (only initialize vars that were not loaded)（仅初始化未加载的vars）
            uninit_vars = []
            for var in tf.global_variables():
                # note - this is hacky, may be better way to do this in newer TF.
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninit_vars.append(var)
            sess.run(tf.variables_initializer(uninit_vars))

            self.start_worker()
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    logger.log("Sampling set of tasks/goals for this meta-batch...") #为这个元批量抽样设置任务/目标

                    env = self.env
                    while 'sample_goals' not in dir(env):
                        env = env.wrapped_env
                    learner_env_goals = env.sample_goals(self.meta_batch_size)

                    self.policy.switch_to_init_dist()  # Switch to pre-update policy切换到更新前的策略

                    all_samples_data, all_paths = [], []
                    for step in range(self.num_grad_updates+1):
                        #if step > 0:
                           #import pdb; pdb.set_trace() # test param_vals functions.
                        logger.log('** Step ' + str(step) + ' **')   #**Step 1**
                        logger.log("Obtaining samples...")     # Obtaining samples...
                        paths = self.obtain_samples(itr, reset_args=learner_env_goals, log_prefix=str(step))# Obtaining samples for iteration 0...

                        all_paths.append(paths)
                        logger.log("Processing samples...")
                        samples_data = {}
                        for key in paths.keys():  # the keys are the tasks
                            # don't log because this will spam the consol with every task.
                            samples_data[key] = self.process_samples(itr, paths[key], log=False)
                        all_samples_data.append(samples_data)
                        # for logging purposes only仅用于记录目的
                        self.process_samples(itr, flatten_list(paths.values()), prefix=str(step), log=True)
                        logger.log("Logging diagnostics...") #记录诊断
                        self.log_diagnostics(flatten_list(paths.values()), prefix=str(step))
                        if step < self.num_grad_updates:
                            logger.log("Computing policy updates...")
                            self.policy.compute_updated_dists(samples_data)


                    logger.log("Optimizing policy...")
                    # This needs to take all samples_data so that it can construct graph for meta-optimization.
                    # 这需要获取所有samples_data，以便它可以构建用于元优化的图形。
                    self.optimize_policy(itr, all_samples_data)
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, all_samples_data[-1])  # , **kwargs)
                    if self.store_paths:
                        params["paths"] = all_samples_data[-1]["paths"]
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")
                    logger.record_tabular('Time', time.time() - start_time)
                    logger.record_tabular('ItrTime', time.time() - itr_start_time)

                    logger.dump_tabular(with_prefix=False)

                    # The rest is some example plotting code.其余的是绘制代码的一些示例。
                    # Plotting code is useful for visualizing trajectories across a few different tasks.绘制代码对于在几个不同任务中可视化轨迹非常有用。
                    if False and itr % 2 == 0 and self.env.observation_space.shape[0] <= 4: # point-mass
                        logger.log("Saving visualization of paths")
                        for ind in range(min(5, self.meta_batch_size)):
                            plt.clf()
                            plt.plot(learner_env_goals[ind][0], learner_env_goals[ind][1], 'k*', markersize=10)
                            plt.hold(True)

                            preupdate_paths = all_paths[0]
                            postupdate_paths = all_paths[-1]

                            pre_points = preupdate_paths[ind][0]['observations']
                            post_points = postupdate_paths[ind][0]['observations']
                            plt.plot(pre_points[:,0], pre_points[:,1], '-r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '-b', linewidth=1)

                            pre_points = preupdate_paths[ind][1]['observations']
                            post_points = postupdate_paths[ind][1]['observations']
                            plt.plot(pre_points[:,0], pre_points[:,1], '--r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '--b', linewidth=1)

                            pre_points = preupdate_paths[ind][2]['observations']
                            post_points = postupdate_paths[ind][2]['observations']
                            plt.plot(pre_points[:,0], pre_points[:,1], '-.r', linewidth=2)
                            plt.plot(post_points[:,0], post_points[:,1], '-.b', linewidth=1)

                            plt.plot(0,0, 'k.', markersize=5)
                            plt.xlim([-0.8, 0.8])
                            plt.ylim([-0.8, 0.8])
                            plt.legend(['goal', 'preupdate path', 'postupdate path'])
                            plt.savefig(osp.join(logger.get_snapshot_dir(), 'prepost_path'+str(ind)+'.png'))
                    elif False and itr % 2 == 0:  # swimmer or cheetah
                        logger.log("Saving visualization of paths")
                        for ind in range(min(5, self.meta_batch_size)):
                            plt.clf()
                            goal_vel = learner_env_goals[ind]
                            plt.title('Swimmer paths, goal vel='+str(goal_vel))
                            plt.hold(True)

                            prepathobs = all_paths[0][ind][0]['observations']
                            postpathobs = all_paths[-1][ind][0]['observations']
                            plt.plot(prepathobs[:,0], prepathobs[:,1], '-r', linewidth=2)
                            plt.plot(postpathobs[:,0], postpathobs[:,1], '--b', linewidth=1)
                            plt.plot(prepathobs[-1,0], prepathobs[-1,1], 'r*', markersize=10)
                            plt.plot(postpathobs[-1,0], postpathobs[-1,1], 'b*', markersize=10)
                            plt.xlim([-1.0, 5.0])
                            plt.ylim([-1.0, 1.0])

                            plt.legend(['preupdate path', 'postupdate path'], loc=2)
                            plt.savefig(osp.join(logger.get_snapshot_dir(), 'swim1d_prepost_itr'+str(itr)+'_id'+str(ind)+'.pdf'))
        self.shutdown_worker()

    def log_diagnostics(self, paths, prefix):
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths)

    def init_opt(self):
        """
        Initialize the optimization procedure. If using tensorflow, this may
        include declaring all the variables and compiling functions
        初始化优化过程。 如果使用tensorflow，这可能包括声明所有变量和编译函数
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr, samples_data):
        """
        Returns all the data that should be saved in the snapshot for this
        iteration.返回应在此次迭代的快照中保存的所有数据。
        """
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        raise NotImplementedError

    def update_plot(self):
        if self.plot:
            plotter.update_plot(self.policy, self.max_path_length)
