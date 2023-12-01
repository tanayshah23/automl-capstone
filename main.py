import os
import sys
import torch
from train import train
from maml import Learner
from dataset import MetaTask
from omegaconf import OmegaConf
from inputs.dataset import get_train_test

"""
This file is used to control the flow based on config
"""
if __name__ == "__main__":
    assert len(sys.argv) > 1, "provide a configuration file"
    config = OmegaConf.load(sys.argv[1])
    os.makedirs(config.output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    print("#" * 40, OmegaConf.to_yaml(config).strip(), "#" * 40, sep="\n")
    train_examples, test_examples = get_train_test(config.data_dir)
    learner = Learner(config.maml)
    test = MetaTask(test_examples, num_task=config.maml.num_test_task, k_support=config.maml.k_support, k_query=config.maml.k_query)
    train(config, learner, train_examples, test)
    torch.save({
        "model_state_dict": learner.model.state_dict(),
        "optimizer_state_dict": learner.outer_optimizer.state_dict()
    }, os.path.join(config.output_dir, "checkpoint.pth"))
    print("LOG: Model checkpoint saved")