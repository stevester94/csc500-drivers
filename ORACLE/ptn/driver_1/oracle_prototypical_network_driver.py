#! /usr/bin/env python3

import numpy as np
import os, json, sys, time

from torch.optim import Adam, optimizer
import torch

from steves_utils.torch_sequential_builder import build_sequential


from steves_models.steves_ptn import Steves_Prototypical_Network
from steves_utils.lazy_iterable_wrapper import Lazy_Iterable_Wrapper
from steves_utils.ptn_train_eval_test_jig import  PTN_Train_Eval_Test_Jig

from steves_utils.torch_utils import confusion_by_domain_over_dataloader

from steves_utils.ORACLE.torch_utils import build_ORACLE_episodic_iterable
from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
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
    base_parameters["device"] = "cuda"

    base_parameters["seed"] = 1337
    base_parameters["desired_serial_numbers"] = [
        "3123D52",
        "3123D65",
        "3123D79",
        "3123D80",
    ]
    base_parameters["source_domains"] = [38]
    base_parameters["target_domains"] = [ 2,
        8,
        14,
        20,
        26,
        32,
        44,
        50,
        56,
        62
    ]

    base_parameters["window_stride"]=50
    base_parameters["window_length"]=256
    base_parameters["desired_runs"]=[1]
    base_parameters["num_examples_per_device"]=75000

    # base_parameters["n_shot"]  = 
    base_parameters["n_query"]  = 10
    base_parameters["n_train_tasks"] = 2000
    base_parameters["n_val_tasks"]  = 1000
    base_parameters["n_test_tasks"]  = 100
    base_parameters["validation_frequency"] = 100

    base_parameters["n_epoch"] = 100

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

    base_parameters["patience"] = 10


experiment_name         = parameters["experiment_name"]
lr                      = parameters["lr"]
# n_epoch                 = parameters["n_epoch"]
# batch_size              = parameters["batch_size"]
# patience                = parameters["patience"]
seed                    = parameters["seed"]
device                  = torch.device(parameters["device"])

desired_serial_numbers  = parameters["desired_serial_numbers"]
source_domains         = parameters["source_domains"]
target_domains         = parameters["target_domains"]
window_stride           = parameters["window_stride"]
window_length           = parameters["window_length"]
desired_runs            = parameters["desired_runs"]
num_examples_per_device = parameters["num_examples_per_device"]

# n_shot        = len(desired_serial_numbers)
n_shot        = 10
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

source_train_dl, source_val_dl, source_test_dl = build_ORACLE_episodic_iterable(
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

target_train_dl, target_val_dl, target_test_dl = build_ORACLE_episodic_iterable(
    desired_serial_numbers=desired_serial_numbers,
    # desired_distances=[50,32,8],
    desired_distances=target_domains,
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

# For CNN We only use X and Y. And we only train on the source.
# Properly form the data using a transform lambda and Lazy_Map. Finally wrap them in a dataloader

transform_lambda = lambda ex: ex[1] # Original is (<domain>, <episode>) so we strip down to episode only

source_train_dl = Lazy_Iterable_Wrapper(source_train_dl, transform_lambda)
source_val_dl = Lazy_Iterable_Wrapper(source_val_dl, transform_lambda)
source_test_dl = Lazy_Iterable_Wrapper(source_test_dl, transform_lambda)
target_train_dl = Lazy_Iterable_Wrapper(target_train_dl, transform_lambda)
target_val_dl = Lazy_Iterable_Wrapper(target_val_dl, transform_lambda)
target_test_dl = Lazy_Iterable_Wrapper(target_test_dl, transform_lambda)



###################################
# Build the model
###################################
model = Steves_Prototypical_Network(x_net)
optimizer = Adam(params=model.parameters(), lr=lr)


###################################
# train
###################################
jig = PTN_Train_Eval_Test_Jig(model, "/tmp/best_model", device)

print("source acc {}, source loss {}".format(*jig.test(source_test_dl)))
print("target acc {}, target loss {}".format(*jig.test(target_test_dl)))

sys.exit(1)



jig.train(
    train_iterable=source_train_dl,
    source_val_iterable=source_val_dl,
    target_val_iterable=target_val_dl,
    num_epochs=n_epoch,
    num_logs_per_epoch=NUM_LOGS_PER_EPOCH,
    patience=patience,
    optimizer=optimizer
)

# train_loss_history = []
# val_loss_history   = []

# best_val_avg_loss = float("inf")
# best_model_state = model.state_dict()
# best_epoch_index = 0

# for epoch in range(n_epoch):
#     train_avg_loss = model.fit(source_train_dl, optimizer, log_frequency=100)
#     val_acc, val_avg_loss = model.validate(source_val_dl)

    
#     train_loss_history.append(train_avg_loss)
#     val_loss_history.append(val_avg_loss)

#     print(f"Val Accuracy: {(100 * val_acc):.2f}%, Val Avg Loss: {val_avg_loss:.2f}")
#     # If this was the best validation performance, we save the model state
#     if val_avg_loss < best_val_avg_loss:
#         print("Best so far")
#         best_model_state = model.state_dict()
#         best_val_avg_loss = val_avg_loss
#         best_epoch_index = epoch
#     elif epoch - best_epoch_index == patience:
#         print("Patience Exhausted")
#         break

# sys.exit(1)


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