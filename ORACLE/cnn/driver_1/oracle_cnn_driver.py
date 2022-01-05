#! /usr/bin/env python3
import os
from numpy.lib.utils import source
import torch
import numpy as np
import os
import sys
import json
import time
from math import floor

from steves_models.configurable_vanilla import Configurable_Vanilla
from steves_utils.vanilla_train_eval_test_jig import  Vanilla_Train_Eval_Test_Jig
from steves_utils.torch_sequential_builder import build_sequential
from steves_utils.lazy_map import Lazy_Map
from steves_utils.sequence_aggregator import Sequence_Aggregator
import steves_utils.ORACLE.torch_utils as ORACLE_Torch
from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
)

from steves_utils.torch_utils import (
    confusion_by_domain_over_dataloader,
)

from steves_utils.utils_v2 import (
    per_domain_accuracy_from_confusion
)

from do_report import do_report


# Parameters relevant to results
RESULTS_DIR = "./results"
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, "loss.png")
EXPERIMENT_JSON_PATH = os.path.join(RESULTS_DIR, "experiment.json")

# Parameters relevant to experiment
NUM_LOGS_PER_EPOCH = 5

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)



###################################
# Parse Args, Set paramaters
###################################
if len(sys.argv) > 1 and sys.argv[1] == "-":
    parameters = json.loads(sys.stdin.read())
elif len(sys.argv) == 1:
    base_parameters = {}
    base_parameters["experiment_name"] = "MANUAL ORACLE CNN"
    base_parameters["lr"] = 0.0001
    base_parameters["n_epoch"] = 10
    base_parameters["batch_size"] = 256
    base_parameters["patience"] = 10
    base_parameters["seed"] = 1337
    base_parameters["dataset_seed"] = 1337
    base_parameters["device"] = "cuda"
    base_parameters["desired_serial_numbers"] = ALL_SERIAL_NUMBERS
    base_parameters["source_domains"] = [50,32,8]
    base_parameters["target_domains"] = list(set(ALL_DISTANCES_FEET) - set([50,32,8]))

    base_parameters["window_stride"]=50
    base_parameters["window_length"]=512
    base_parameters["desired_runs"]=[1]
    base_parameters["num_examples_per_device_per_domain"]=1000
    base_parameters["max_cache_items"] = 4.5e6

    base_parameters["criteria_for_best"] = "source"



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

        {"class": "Linear", "kargs": {"in_features": 256, "out_features": len(base_parameters["desired_serial_numbers"])}},
    ]

    parameters = base_parameters


experiment_name = parameters["experiment_name"]
lr              = parameters["lr"]
n_epoch         = parameters["n_epoch"]
batch_size      = parameters["batch_size"]
patience        = parameters["patience"]
seed            = parameters["seed"]
dataset_seed    = parameters["dataset_seed"]
device          = torch.device(parameters["device"])

desired_serial_numbers             = parameters["desired_serial_numbers"]
source_domains                     = parameters["source_domains"]
target_domains                     = parameters["target_domains"]
window_stride                      = parameters["window_stride"]
window_length                      = parameters["window_length"]
desired_runs                       = parameters["desired_runs"]
num_examples_per_device_per_domain = parameters["num_examples_per_device_per_domain"]
max_cache_items                    = int(parameters["max_cache_items"])
criteria_for_best                  = parameters["criteria_for_best"]


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
print("Building source dataset")

# The num_examples_per_device looks a little weird because we pull N examples randomly from the desired domains
# So this is roughly correct
source_ds = ORACLE_Torch.ORACLE_Torch_Dataset(
                desired_serial_numbers=desired_serial_numbers,
                desired_distances=source_domains,
                desired_runs=desired_runs,
                window_length=window_length,
                window_stride=window_stride,
                num_examples_per_device=num_examples_per_device_per_domain*len(source_domains),
                seed=dataset_seed,  
                max_cache_size=max_cache_items,
                transform_func=lambda x: (x["iq"], serial_number_to_id(x["serial_number"]), x["distance_ft"]),
                prime_cache=False
)

print("Build target dataset")

target_ds = ORACLE_Torch.ORACLE_Torch_Dataset(
                desired_serial_numbers=desired_serial_numbers,
                desired_distances=target_domains,
                desired_runs=desired_runs,
                window_length=window_length,
                window_stride=window_stride,
                num_examples_per_device=num_examples_per_device_per_domain*len(target_domains),
                seed=dataset_seed,  
                max_cache_size=max_cache_items,
                transform_func=lambda x: (x["iq"], serial_number_to_id(x["serial_number"]), x["distance_ft"]),
                prime_cache=False
)


def wrap_in_dataloader(ds):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=50,
        pin_memory=True
    )


# Split our source and target datasets into train val and test
source_train_len = floor(len(source_ds)*0.7)
source_val_len   = floor(len(source_ds)*0.15)
source_test_len  = len(source_ds) - source_train_len - source_val_len
source_train_ds, source_val_ds, source_test_ds = torch.utils.data.random_split(source_ds, [source_train_len, source_val_len, source_test_len], generator=torch.Generator().manual_seed(seed))


target_train_len = floor(len(target_ds)*0.7)
target_val_len   = floor(len(target_ds)*0.15)
target_test_len  = len(target_ds) - target_train_len - target_val_len
target_train_ds, target_val_ds, target_test_ds = torch.utils.data.random_split(target_ds, [target_train_len, target_val_len, target_test_len], generator=torch.Generator().manual_seed(seed))

# For CNN We only use X and Y. And we only train on the source.
# Properly form the data using a transform lambda and Lazy_Map. Finally wrap them in a dataloader

transform_lambda = lambda ex: ex[:2] # Strip the tuple to just (x,y)

# CIDA combines source and target training sets into a single dataloader, that's why this one is just called train_dl
train_dl = wrap_in_dataloader(
    Lazy_Map(source_train_ds, transform_lambda)
)

source_val_dl = wrap_in_dataloader(
    Lazy_Map(source_val_ds, transform_lambda)
)
source_test_dl = wrap_in_dataloader(
    Lazy_Map(source_test_ds, transform_lambda)
)

target_val_dl = wrap_in_dataloader(
    Lazy_Map(target_val_ds, transform_lambda)
)
target_test_dl  = wrap_in_dataloader(
    Lazy_Map(target_test_ds, transform_lambda)
)


###################################
# Build the model
###################################
model = Configurable_Vanilla(
    x_net=x_net,
    label_loss_object=torch.nn.NLLLoss(),
    learning_rate=lr
)


###################################
# Build the tet jig, train
###################################
jig = Vanilla_Train_Eval_Test_Jig(
    model=model,
    path_to_best_model=BEST_MODEL_PATH,
    device=device,
    label_loss_object=torch.nn.NLLLoss(),
)

jig.train(
    train_iterable=train_dl,
    source_val_iterable=source_val_dl,
    target_val_iterable=target_val_dl,
    patience=patience,
    num_epochs=n_epoch,
    num_logs_per_epoch=NUM_LOGS_PER_EPOCH,
    criteria_for_best=criteria_for_best
)


###################################
# Evaluate the model
###################################
source_test_label_accuracy, source_test_label_loss = jig.test(source_test_dl)
target_test_label_accuracy, target_test_label_loss = jig.test(target_test_dl)

source_val_label_accuracy, source_val_label_loss = jig.test(source_val_dl)
target_val_label_accuracy, target_val_label_loss = jig.test(target_val_dl)

history = jig.get_history()

total_epochs_trained = len(history["epoch_indices"])

val_dl = wrap_in_dataloader(Sequence_Aggregator((source_val_ds, target_val_ds)))

confusion = confusion_by_domain_over_dataloader(model, device, val_dl, forward_uses_domain=False)
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



print("Source Test Label Accuracy:", source_test_label_accuracy, "Target Test Label Accuracy:", target_test_label_accuracy)
print("Source Val Label Accuracy:", source_val_label_accuracy, "Target Val Label Accuracy:", target_val_label_accuracy)

with open(EXPERIMENT_JSON_PATH, "w") as f:
    json.dump(experiment, f, indent=2)


###################################
# Make the report
###################################
do_report(EXPERIMENT_JSON_PATH, LOSS_CURVE_PATH)
