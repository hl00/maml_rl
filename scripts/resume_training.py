


from rllab.sampler.utils import rollout
from rllab.algos.batch_polopt import BatchPolopt
import argparse     #argparse是一个命令解析器
import joblib
import uuid
import os
import random
import numpy as np
import json
import subprocess
from rllab.misc import logger
from rllab.misc.instrument import to_local_command
filename = str(uuid.uuid4()) #uuid.uuid4() : 基于随机数生成uuid;str() 函数返回一个对象的string格式

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    #创建一个命令解析器的句柄
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')     #add_argument:读入命令行参数，该调用有多个参数
    parser.add_argument('--log_dir', type=str, default=None,
                        help='path to the new log directory')   #对于函数add_argumen()第一个是选项，第二个是数据类型，第三个默认值，第四个是help命令时的说明
    # Look for params.json file
    args = parser.parse_args()
    parent_dir = os.path.dirname(os.path.realpath(args.file))
    json_file_path = os.path.join(parent_dir, "params.json")
    logger.log("Looking for params.json at %s..." % json_file_path)
    try:
        with open(json_file_path, "r") as f:
            params = json.load(f)
        # exclude certain parameters
        excluded = ['json_args']
        for k in excluded:
            if k in params:
                del params[k]
        for k, v in list(params.items()):
            if v is None:
                del params[k]
        if args.log_dir is not None:
            params['log_dir'] = args.log_dir
        params['resume_from'] = args.file
        command = to_local_command(params, script='scripts/run_experiment_lite.py')
        print(command)
        try:
            subprocess.call(command, shell=True, env=os.environ)
        except Exception as e:
            print(e)
            if isinstance(e, KeyboardInterrupt):
                raise
    except IOError as e:
        logger.log("Failed to find json file. Continuing in non-stub mode...")
        data = joblib.load(args.file)
        assert 'algo' in data
        algo = data['algo']
        assert isinstance(algo, BatchPolopt)
        algo.train()
