#! /usr/bin/env python3

import easydict
import numpy as np
import os, json, time, sys, random
from easydict import EasyDict

from torch.optim import Adam
import torch


from steves_utils.torch_sequential_builder import build_sequential


from steves_models.steves_ptn import Steves_Prototypical_Network
from steves_utils.lazy_iterable_wrapper import Lazy_Iterable_Wrapper
from steves_utils.iterable_aggregator import Iterable_Aggregator
from steves_utils.ptn_train_eval_test_jig import  PTN_Train_Eval_Test_Jig

from steves_utils.torch_utils import ptn_confusion_by_domain_over_dataloader
from steves_utils.utils_v2 import per_domain_accuracy_from_confusion

from steves_utils.CORES.utils import (
    ALL_NODES,
    build_CORES_episodic_iterable,
    ALL_NODES_MINIMUM_1000_EXAMPLES,
    ALL_DAYS
)

from do_report import do_report




###################################
# Default Arguments
###################################

base_parameters = {}
base_parameters["experiment_name"] = "MANUAL CORES PTN"
base_parameters["lr"] = 0.001
base_parameters["device"] = "cuda"

base_parameters["seed"] = 1337
base_parameters["dataset_seed"] = 1337
base_parameters["desired_classes_source"] = ALL_NODES_MINIMUM_1000_EXAMPLES
# base_parameters["desired_classes_target"] = list(set(ALL_NODES) - set(ALL_NODES_MINIMUM_1000_EXAMPLES))
base_parameters["desired_classes_target"] = ALL_NODES_MINIMUM_1000_EXAMPLES

base_parameters["source_domains"] = [1]
base_parameters["target_domains"] = [2,3,4,5]

base_parameters["num_examples_per_class_per_domain_source"]=100
base_parameters["num_examples_per_class_per_domain_target"]=100

base_parameters["n_shot"] = 2
base_parameters["n_way"]  = len(base_parameters["desired_classes_source"])
base_parameters["n_query"]  = 1
base_parameters["train_k_factor"] = 1
base_parameters["val_k_factor"] = 2
base_parameters["test_k_factor"] = 2

base_parameters["n_epoch"] = 3

base_parameters["patience"] = 10

base_parameters["criteria_for_best"] = "target"

base_parameters["x_net"] =     [
    {"class": "nnReshape", "kargs": {"shape":[-1, 1, 2, 256]}},
    {"class": "Conv2d", "kargs": { "in_channels":1, "out_channels":256, "kernel_size":(1,7), "bias":False, "padding":(0,3), },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":256}},

    {"class": "Conv2d", "kargs": { "in_channels":256, "out_channels":80, "kernel_size":(2,7), "bias":True, "padding":(0,3), },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":80}},
    {"class": "Flatten", "kargs": {}},

    {"class": "Linear", "kargs": {"in_features": 80*256, "out_features": 256}}, # 80 units per IQ pair
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm1d", "kargs": {"num_features":256}},

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
]


# Parameters relevant to results
# These parameters will basically never need to change
base_parameters["NUM_LOGS_PER_EPOCH"] = 10
base_parameters["RESULTS_DIR"] = "./results"
base_parameters["EXPERIMENT_JSON_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "experiment.json")
base_parameters["LOSS_CURVE_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "loss.png")
base_parameters["BEST_MODEL_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "best_model.pth")





def parse_and_validate_parameters(d:dict)->EasyDict:
    ed = EasyDict(d)
    allowed_keys =set(base_parameters.keys())
    supplied_keys = set(ed.keys())

    

    if  supplied_keys != allowed_keys:
        print("Parameters are incorrect")
        if len(supplied_keys - allowed_keys)>0: print("Shouldn't have:", str(supplied_keys - allowed_keys))
        if len(allowed_keys - supplied_keys)>0: print("Need to have:", str(allowed_keys - supplied_keys))
        raise RuntimeError("Parameters are incorrect")

    return ed


"""
# Simple pass-through to the results json
experiment_name         = parameters["experiment_name"]

# Learning rate for Adam optimizer
lr                      = parameters["lr"]

# Sets seed for anything that uses a seed. Allows the experiment to be perfectly reproducible
seed                    = parameters["seed"]
dataset_seed            = parameters["dataset_seed"]

# Which device we run on ['cpu', 'cuda']
# Note for PTN this must be 'cuda'
device                  = torch.device(parameters["device"])

# Radio devices (nodes)
desired_classes_source  = parameters["desired_classes_source"]
desired_classes_target  = parameters["desired_classes_target"]

# Days
source_domains         = parameters["source_domains"]

# Days
target_domains         = parameters["target_domains"]

num_examples_per_class_per_domain_source = parameters["num_examples_per_class_per_domain_source"]
num_examples_per_class_per_domain_target = parameters["num_examples_per_class_per_domain_target"]

train_k_factor = parameters["train_k_factor"]
val_k_factor   = parameters["val_k_factor"]
test_k_factor  = parameters["test_k_factor"]

# The n_way of the episodes. Prior literature suggests keeping
# this consistent between train and test. I suggest keeping this
# == to the number of labels but that is not a hard and fast rule
n_way         = parameters["n_way"]

# The number of examples per class in the support set in each episode
n_shot        = parameters["n_shot"]

# The number of examples per class in the query set for each epsidode
n_query       = parameters["n_query"]



# The maximumum number of "epochs" to train. Note that an epoch is simply a full
# iteration of the training dataset, it absolutely does not imply that we have iterated
# over every training example available
n_epoch = parameters["n_epoch"]

# How many epochs to train before giving up due to no improvement in loss.
# Note that patience for PTN considers source_val_loss + target_val_loss.
patience = parameters["patience"]

# How to pick the best model (and when to give up training)
# source, target, source_and_target
criteria_for_best = parameters["criteria_for_best"]

# A list of dictionaries representation of a sequential neural network.
# The network gets instantiated by my custom 'build_sequential' function.
# The args and class types are typically a straight pass through to the 
# corresponding torch layers
parameters["x_net"]
"""

###################################
# Set the RNGs and make it all deterministic
###################################
def set_rng(p:EasyDict)->None:
    np.random.seed(p.seed)
    random.seed(p.seed)
    torch.manual_seed(p.seed)

    torch.use_deterministic_algorithms(True) 




###################################
# Build the network(s)
# Note: It's critical to do this AFTER setting the RNG
###################################
def build_network(p:EasyDict):
    return build_sequential(p.x_net)



###################################
# Build the dataset
###################################
def build_datasets(p:EasyDict)->EasyDict:
    """
    returns {
        "source": {
            "original": {"train":<data>, "val":<data>, "test":<data>}
            "processed": {"train":<data>, "val":<data>, "test":<data>}
        },
        "target": {
            "original": {"train":<data>, "val":<data>, "test":<data>}
            "processed": {"train":<data>, "val":<data>, "test":<data>}
        },
    }
    """
    source_original_train, source_original_val, source_original_test = build_CORES_episodic_iterable(
        days_to_get=p.source_domains,
        n_query=p.n_query,
        n_way=p.n_way,
        n_shot=p.n_shot,
        nodes_to_get=p.desired_classes_source,
        num_examples_per_node_per_day=p.num_examples_per_class_per_domain_source,
        dataset_seed=p.dataset_seed,
        iterator_seed=p.seed,
        train_k_factor=p.train_k_factor,
        val_k_factor=p.val_k_factor,
        test_k_factor=p.test_k_factor,
    ) 

    target_original_train, target_original_val, target_original_test = build_CORES_episodic_iterable(
        days_to_get=p.target_domains,
        n_query=p.n_query,
        n_way=p.n_way,
        n_shot=p.n_shot,
        nodes_to_get=p.desired_classes_target,
        num_examples_per_node_per_day=p.num_examples_per_class_per_domain_target,
        dataset_seed=p.dataset_seed,
        iterator_seed=p.seed,
        train_k_factor=p.train_k_factor,
        val_k_factor=p.val_k_factor,
        test_k_factor=p.test_k_factor,
    )

    # For CNN We only use X and Y. And we only train on the source.
    # Properly form the data using a transform lambda and Lazy_Iterable_Wrapper. Finally wrap them in a dataloader

    transform_lambda = lambda ex: ex[1] # Original is (<domain>, <episode>) so we strip down to episode only

    source_processed_train = Lazy_Iterable_Wrapper(source_original_train, transform_lambda)
    source_processed_val   = Lazy_Iterable_Wrapper(source_original_val, transform_lambda)
    source_processed_test  = Lazy_Iterable_Wrapper(source_original_test, transform_lambda)

    target_processed_train = Lazy_Iterable_Wrapper(target_original_train, transform_lambda)
    target_processed_val   = Lazy_Iterable_Wrapper(target_original_val, transform_lambda)
    target_processed_test  = Lazy_Iterable_Wrapper(target_original_test, transform_lambda)



    return EasyDict({
        "source": {
            "original": {"train":source_original_train, "val":source_original_val, "test":source_original_test},
            "processed": {"train":source_processed_train, "val":source_processed_val, "test":source_processed_test}
        },
        "target": {
            "original": {"train":target_original_train, "val":target_original_val, "test":target_original_test},
            "processed": {"train":target_processed_train, "val":target_processed_val, "test":target_processed_test}
        },
    })





###################################
# Build the model
###################################
def build_model(p:EasyDict)->tuple:
    model = Steves_Prototypical_Network(x_net, x_shape=(2,256))
    optimizer = Adam(params=model.parameters(), lr=p.lr)
    return model, optimizer


###################################
# train
###################################
def train(p:easydict, model, optimizer, datasets:EasyDict)->PTN_Train_Eval_Test_Jig:
    jig = PTN_Train_Eval_Test_Jig(model, p.BEST_MODEL_PATH, p.device)

    jig.train(
        train_iterable=datasets.source.processed.train,
        source_val_iterable=datasets.source.processed.val,
        target_val_iterable=datasets.target.processed.val,
        num_epochs=p.n_epoch,
        num_logs_per_epoch=p.NUM_LOGS_PER_EPOCH,
        patience=p.patience,
        optimizer=optimizer,
        criteria_for_best=p.criteria_for_best,
    )

    return jig


###################################
# Evaluate the model
###################################
def evaluate_model_and_create_experiment_summary(
    p:EasyDict,
    jig:PTN_Train_Eval_Test_Jig,
    total_experiment_time_secs,
    ds:EasyDict
    )->dict:
    source_test_label_accuracy, source_test_label_loss = jig.test(ds.source.processed.test)
    target_test_label_accuracy, target_test_label_loss = jig.test(ds.target.processed.test)

    source_val_label_accuracy, source_val_label_loss = jig.test(ds.source.processed.val)
    target_val_label_accuracy, target_val_label_loss = jig.test(ds.target.processed.val)

    history = jig.get_history()

    total_epochs_trained = len(history["epoch_indices"])

    val_dl = Iterable_Aggregator((ds.source.original.val,ds.target.original.val))

    confusion = ptn_confusion_by_domain_over_dataloader(model, p.device, val_dl)
    per_domain_accuracy = per_domain_accuracy_from_confusion(confusion)

    # Add a key to per_domain_accuracy for if it was a source domain
    for domain, accuracy in per_domain_accuracy.items():
        per_domain_accuracy[domain] = {
            "accuracy": accuracy,
            "source?": domain in p.source_domains
        }


    experiment = {
        "experiment_name": p.experiment_name,
        "parameters": dict(p),
        "results": {
            "source_test_label_accuracy": source_test_label_accuracy,
            "source_test_label_loss": source_test_label_loss,
            "target_test_label_accuracy": target_test_label_accuracy,
            "target_test_label_loss": target_test_label_loss,
            "source_val_label_accuracy": source_val_label_accuracy,
            "source_val_label_loss": source_val_label_loss,
            "target_val_label_accuracy": target_val_label_accuracy,
            "target_val_label_loss": target_val_label_loss,
            "total_epochs_trained": total_epochs_trained,
            "total_experiment_time_secs": total_experiment_time_secs,
            "confusion": confusion,
            "per_domain_accuracy": per_domain_accuracy,
        },
        "history": history,
    }

    return experiment

    




###################################
# Write out the results
###################################
def write_results(p:EasyDict, experiment)->None:
    with open(p.EXPERIMENT_JSON_PATH, "w") as f:
        json.dump(experiment, f, indent=2)



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-":
        parameters = json.loads(sys.stdin.read())
    elif len(sys.argv) == 1:
        parameters = base_parameters
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Unit Testing!")
        raise RuntimeError("Unimplemented")

    # Required since we're pulling in 3rd party code
    torch.set_default_dtype(torch.float64)

    start_time_secs = time.time()

    p = parse_and_validate_parameters(parameters)

    if not os.path.exists(p.RESULTS_DIR):
        os.mkdir(p.RESULTS_DIR)

    set_rng(p)
    x_net = build_network(p)
    datasets = build_datasets(p)
    model, opt = build_model(p)
    jig = train(
        p,
        model=model,
        optimizer=opt,
        datasets=datasets
    )
    total_experiment_time_secs = time.time() - start_time_secs

    experiment = evaluate_model_and_create_experiment_summary(
        p,
        jig=jig,
        ds=datasets,
        total_experiment_time_secs=total_experiment_time_secs,
    )

    print("Source Test Label Accuracy:", experiment["results"]["source_test_label_accuracy"], "Target Test Label Accuracy:", experiment["results"]["target_test_label_accuracy"])
    print("Source Val Label Accuracy:", experiment["results"]["source_val_label_accuracy"], "Target Val Label Accuracy:", experiment["results"]["target_val_label_accuracy"])

    write_results(p, experiment)

    do_report(p.EXPERIMENT_JSON_PATH, p.LOSS_CURVE_PATH)