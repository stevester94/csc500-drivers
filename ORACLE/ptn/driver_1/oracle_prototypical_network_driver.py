#! /usr/bin/env python3

import numpy as np
import os, json, sys, time

from torch.optim import Adam
import torch

from steves_utils.torch_sequential_builder import build_sequential


from steves_models.steves_ptn import Steves_Prototypical_Network
from steves_utils.lazy_iterable_wrapper import Lazy_Iterable_Wrapper
from steves_utils.ORACLE.torch_utils import build_ORACLE_episodic_iterable
from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
)

from do_report import do_report

MAX_CACHE_SIZE = int(4e9)

# Required since we're pulling in 3rd party code
torch.set_default_dtype(torch.float64)


# Parameters relevant to results
RESULTS_DIR = "./results"
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
EXPERIMENT_JSON_PATH = os.path.join(RESULTS_DIR, "experiment.json")
LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, "loss.png")


###################################
# Parse Args, Set paramaters
###################################
if len(sys.argv) > 1 and sys.argv[1] == "-":
    parameters = json.loads(sys.stdin.read())
elif len(sys.argv) == 1:
    base_parameters = {}
    base_parameters["experiment_name"] = "One Distance ORACLE CNN"
    base_parameters["lr"] = 0.001
    # base_parameters["n_epoch"] = 10
    # base_parameters["batch_size"] = 256
    # base_parameters["patience"] = 10
    base_parameters["seed"] = 1337
    # base_parameters["device"] = "cuda"
    base_parameters["desired_serial_numbers"] = [
        "3123D52",
        "3123D65",
        "3123D79",
        "3123D80",
    ]
    base_parameters["source_domains"] = [2]
    # base_parameters["target_domains"] = list(set(ALL_DISTANCES_FEET) - set([50,32,8]))

    base_parameters["window_stride"]=50
    base_parameters["window_length"]=256
    base_parameters["desired_runs"]=[1]
    base_parameters["num_examples_per_device"]=75000

    # base_parameters["n_shot"]  = 
    base_parameters["n_query"]  = 10
    base_parameters["n_train_tasks"] = 500
    base_parameters["n_val_tasks"]  = 100
    base_parameters["n_test_tasks"]  = 100
    base_parameters["validation_frequency"] = 100

    base_parameters["n_epoch"] = 5

    base_parameters["x_net"] =     [# droupout, groups, 512 out
        {"class": "nnReshape", "kargs": {"shape":[-1, 1, 2, 128]}},
        {"class": "Conv2d", "kargs": { "in_channels":1, "out_channels":256, "kernel_size":(1,7), "bias":False, "padding":(0,3), },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "BatchNorm2d", "kargs": {"num_features":256}},

        {"class": "Conv2d", "kargs": { "in_channels":256, "out_channels":80, "kernel_size":(2,7), "bias":True, "padding":(0,3), },},
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "BatchNorm2d", "kargs": {"num_features":80}},
        {"class": "Flatten", "kargs": {}},

        {"class": "Linear", "kargs": {"in_features": 80*128, "out_features": 256}}, # 80 units per IQ pair
        {"class": "ReLU", "kargs": {"inplace": True}},
        {"class": "BatchNorm1d", "kargs": {"num_features":256}},

        {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
    ]


    parameters = base_parameters

    base_parameters["patience"] = 2


experiment_name         = parameters["experiment_name"]
lr                      = parameters["lr"]
# n_epoch                 = parameters["n_epoch"]
# batch_size              = parameters["batch_size"]
# patience                = parameters["patience"]
seed                    = parameters["seed"]

desired_serial_numbers  = parameters["desired_serial_numbers"]
source_domains         = parameters["source_domains"]
# target_domains         = parameters["target_domains"]
window_stride           = parameters["window_stride"]
window_length           = parameters["window_length"]
desired_runs            = parameters["desired_runs"]
num_examples_per_device = parameters["num_examples_per_device"]

n_shot        = len(desired_serial_numbers)
n_query       = parameters["n_query"]
n_train_tasks = parameters["n_train_tasks"]
n_val_tasks   = parameters["n_val_tasks"]
n_test_tasks  = parameters["n_test_tasks"]

validation_frequency = parameters["validation_frequency"]
n_epoch = parameters["n_epoch"]
patience = parameters["patience"]

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

train_dl, val_dl, test_dl = build_ORACLE_episodic_iterable(
    desired_serial_numbers=desired_serial_numbers,
    # desired_distances=[50,32,8],
    desired_distances=source_domains,
    desired_runs=desired_runs,
    window_length=window_length,
    window_stride=window_stride,
    num_examples_per_device=num_examples_per_device,
    seed=seed,
    max_cache_size=MAX_CACHE_SIZE,
    n_way=len(desired_serial_numbers),
    n_shot=n_shot,
    n_query=n_query,
    n_train_tasks_per_distance=n_train_tasks,
    n_val_tasks_per_distance=n_val_tasks,
    n_test_tasks_per_distance=n_test_tasks,
)

lam = lambda k: k[1]
train_dl = Lazy_Iterable_Wrapper(train_dl , lam)
val_dl = Lazy_Iterable_Wrapper(val_dl , lam)
test_dl = Lazy_Iterable_Wrapper(test_dl , lam)

###################################
# Build the model
###################################
model = Steves_Prototypical_Network(x_net).cuda()
optimizer = Adam(params=model.parameters(), lr=lr)




###################################
# train
###################################
train_loss_history = []
val_loss_history   = []

best_val_avg_loss = float("inf")
best_model_state = model.state_dict()
best_epoch_index = 0

for epoch in range(n_epoch):
    train_avg_loss = model.fit(train_dl, optimizer, log_frequency=100)
    val_acc, val_avg_loss = model.validate(val_dl)

    
    train_loss_history.append(train_avg_loss)
    val_loss_history.append(val_avg_loss)

    print(f"Val Accuracy: {(100 * val_acc):.2f}%, Val Avg Loss: {val_avg_loss:.2f}")
    # If this was the best validation performance, we save the model state
    if val_avg_loss < best_val_avg_loss:
        print("Best so far")
        best_model_state = model.state_dict()
        best_val_avg_loss = val_avg_loss
        best_epoch_index = epoch
    elif epoch - best_epoch_index == patience:
        print("Patience Exhausted")
        break



###################################
# evaluate
###################################
print("Reloading best model")
model.load_state_dict(best_model_state)
val_accuracy, val_loss = model.evaluate(val_dl)

print(f"Validation Accuracy: {100 * val_accuracy:.2f}%")

total_experiment_time_secs = time.time() - start_time_secs


###################################
# save results
###################################
experiment = {
    "experiment_name": experiment_name,
    "parameters": parameters,
    "val_accuracy":val_accuracy,
    "train_loss_history":train_loss_history,
    "val_loss_history":val_loss_history,
    "total_experiment_time_secs":total_experiment_time_secs,
}



with open(EXPERIMENT_JSON_PATH, "w") as f:
    json.dump(experiment, f, indent=2)


do_report(EXPERIMENT_JSON_PATH, LOSS_CURVE_PATH)

# NUM_EPOCHS = 25

# for epoch in range(NUM_EPOCHS):
#     epoch_train_loss = model.fit(train_dl, optimizer, val_loader=val_dl, validation_frequency=500)
#     accuracy = model.evaluate(test_dl)

#     print(epoch_train_loss)

#     print(f"Average Val Accuracy : {(100 * accuracy):.2f}")
#     print(f"Average Loss: {(epoch_train_loss):.2f}")