#! /usr/bin/env python3

import numpy as np
import os, json, sys, time

from torch.optim import Adam, optimizer
import torch

from tqdm import tqdm

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

MAX_CACHE_SIZE = int(4e9)
NUM_LOGS_PER_EPOCH = 10

# Required since we're pulling in 3rd party code
torch.set_default_dtype(torch.float64)


# Parameters relevant to results
RESULTS_DIR = "./results"
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
EXPERIMENT_JSON_PATH = os.path.join(RESULTS_DIR, "experiment.json")
LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, "loss.png")
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")



###################################
# Parse Args, Set paramaters
###################################
if len(sys.argv) > 1 and sys.argv[1] == "-":
    parameters = json.loads(sys.stdin.read())
elif len(sys.argv) == 1:
    base_parameters = {}
    base_parameters["experiment_name"] = "MANUAL CORES PTN"
    base_parameters["lr"] = 0.001
    base_parameters["device"] = "cuda"

    base_parameters["seed"] = 1337
    base_parameters["dataset_seed"] = 1337
    base_parameters["desired_classes"] = ALL_NODES_MINIMUM_1000_EXAMPLES
    base_parameters["desired_classes"] = ALL_NODES

    base_parameters["source_domains"] = [1,2]
    base_parameters["target_domains"] = [3,4,5]

    base_parameters["source_domains"] = [1]
    base_parameters["target_domains"] = [2,3,4,5]

    base_parameters["num_examples_per_class_per_domain"]=100

    base_parameters["n_shot"] = 3
    base_parameters["n_way"]  = len(base_parameters["desired_classes"])
    base_parameters["n_query"]  = 2
    base_parameters["train_k_factor"] = 1
    base_parameters["val_k_factor"] = 1
    base_parameters["test_k_factor"] = 1

    base_parameters["n_epoch"] = 3

    base_parameters["patience"] = 10


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


    parameters = base_parameters


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
desired_classes  = parameters["desired_classes"]

# Distances
source_domains         = parameters["source_domains"]

# Distances
target_domains         = parameters["target_domains"]

num_examples_per_class_per_domain = parameters["num_examples_per_class_per_domain"]

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

# A list of dictionaries representation of a sequential neural network.
# The network gets instantiated by my custom 'build_sequential' function.
# The args and class types are typically a straight pass through to the 
# corresponding torch layers
parameters["x_net"]


start_time_secs = time.time()




###################################
# Set the RNGs and make it all deterministic
###################################
import random 
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

torch.use_deterministic_algorithms(True) 


###################################
# Build the network(s)
# Note: It's critical to do this AFTER setting the RNG
###################################
x_net           = build_sequential(parameters["x_net"])

###################################
# Build the dataset
###################################
print("Building dataset...")
og_source_train, og_source_val, og_source_test = build_CORES_episodic_iterable(
    days_to_get=source_domains,
    n_query=n_query,
    n_way=n_way,
    n_shot=n_shot,
    nodes_to_get=desired_classes,
    num_examples_per_node_per_day=num_examples_per_class_per_domain,
    seed=dataset_seed,
    train_k_factor=train_k_factor,
    val_k_factor=val_k_factor,
    test_k_factor=test_k_factor,
) 

og_target_train, og_target_val, og_target_test = build_CORES_episodic_iterable(
    days_to_get=target_domains,
    n_query=n_query,
    n_way=n_way,
    n_shot=n_shot,
    nodes_to_get=desired_classes,
    num_examples_per_node_per_day=num_examples_per_class_per_domain,
    seed=dataset_seed,
    train_k_factor=train_k_factor,
    val_k_factor=val_k_factor,
    test_k_factor=test_k_factor,
)


def wrap_in_dataloader(ds:list):
    return ds
    return torch.utils.data.DataLoader(
        ds, 
        batch_size=None, # episodic iterable is basically a batch
        shuffle=False,
        num_workers=0,
        # persistent_workers=True,
        # prefetch_factor=5,
        pin_memory=True
    )
    return torch.utils.data.DataLoader(
        ds, 
        batch_size=None, # episodic iterable is basically a batch
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=5,
        pin_memory=True
    )


og_source_train_dl, og_source_val_dl, og_source_test_dl = [wrap_in_dataloader(l) for l in (og_source_train, og_source_val, og_source_test)]
og_target_train_dl, og_target_val_dl, og_target_test_dl = [wrap_in_dataloader(l) for l in (og_target_train, og_target_val, og_target_test)]

# For CNN We only use X and Y. And we only train on the source.
# Properly form the data using a transform lambda and Lazy_Iterable_Wrapper. Finally wrap them in a dataloader

transform_lambda = lambda ex: ex[1] # Original is (<domain>, <episode>) so we strip down to episode only

source_train_dl = Lazy_Iterable_Wrapper(og_source_train_dl, transform_lambda)
source_val_dl = Lazy_Iterable_Wrapper(og_source_val_dl, transform_lambda)
source_test_dl = Lazy_Iterable_Wrapper(og_source_test_dl, transform_lambda)
target_train_dl = Lazy_Iterable_Wrapper(og_target_train_dl, transform_lambda)
target_val_dl = Lazy_Iterable_Wrapper(og_target_val_dl, transform_lambda)
target_test_dl = Lazy_Iterable_Wrapper(og_target_test_dl, transform_lambda)


# x = next(iter(source_train_dl))[0][0]
# x = torch.from_numpy(np.ones([2,128]))
# x = x.reshape(1,2,128)
# print(x.shape)
# print(x_net(x).shape)
# sys.exit(1)


###################################
# Build the model
###################################
model = Steves_Prototypical_Network(x_net, x_shape=(2,256))
optimizer = Adam(params=model.parameters(), lr=lr)


###################################
# train
###################################
jig = PTN_Train_Eval_Test_Jig(model, BEST_MODEL_PATH, device)

print("Begin training")
jig.train(
    train_iterable=source_train_dl,
    source_val_iterable=source_val_dl,
    target_val_iterable=target_val_dl,
    num_epochs=n_epoch,
    num_logs_per_epoch=NUM_LOGS_PER_EPOCH,
    patience=patience,
    optimizer=optimizer,
    criteria_for_best="source_and_target",
)

# model.load_state_dict(torch.load(BEST_MODEL_PATH))


###################################
# Evaluate the model
###################################

source_test_label_accuracy, source_test_label_loss = jig.test(source_test_dl)
target_test_label_accuracy, target_test_label_loss = jig.test(target_test_dl)

source_val_label_accuracy, source_val_label_loss = jig.test(source_val_dl)
target_val_label_accuracy, target_val_label_loss = jig.test(target_val_dl)

history = jig.get_history()

total_epochs_trained = len(history["epoch_indices"])

val_dl = Iterable_Aggregator((og_source_val_dl,og_target_val_dl))

confusion = ptn_confusion_by_domain_over_dataloader(model, device, val_dl)
per_domain_accuracy = per_domain_accuracy_from_confusion(confusion)

# Add a key to per_domain_accuracy for if it was a source domain
for domain, accuracy in per_domain_accuracy.items():
    per_domain_accuracy[domain] = {
        "accuracy": accuracy,
        "source?": domain in source_domains
    }

total_experiment_time_secs = time.time() - start_time_secs

###################################
# Write out the results
###################################

experiment = {
    "experiment_name": experiment_name,
    "parameters": parameters,
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

# print(experiment)

print("Source Test Label Accuracy:", source_test_label_accuracy, "Target Test Label Accuracy:", target_test_label_accuracy)
print("Source Val Label Accuracy:", source_val_label_accuracy, "Target Val Label Accuracy:", target_val_label_accuracy)

with open(EXPERIMENT_JSON_PATH, "w") as f:
    json.dump(experiment, f, indent=2)


###################################
# Make the report
###################################
do_report(EXPERIMENT_JSON_PATH, LOSS_CURVE_PATH)
