#! /usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.gridspec
import json
from steves_utils.vanilla_train_eval_test_jig import Vanilla_Train_Eval_Test_Jig
import pandas as pds
import matplotlib.patches as mpatches

from steves_utils.utils_v2 import do_graph

def do_report(experiment_json_path, loss_curve_path, show_only=False):

    with open(experiment_json_path) as f:
        experiment = json.load(f)

    fig, axes = plt.subplots(1, 1)
    plt.tight_layout()

    fig.suptitle("Experiment Summary")
    fig.set_size_inches(30, 15)

    plt.subplots_adjust(hspace=0.4)
    plt.rcParams['figure.dpi'] = 163

    ###
    # Get Loss Curve
    ###
    graphs = [
            {
            "x": range(len(experiment["train_loss_history"])),
            "y": experiment["train_loss_history"],
            "x_label": None,
            "y_label": "Train Label Loss",
            "x_units": "Epoch",
            "y_units": None,
            }, 
            {
            "x": range(len(experiment["train_loss_history"])),
            "y": experiment["val_loss_history"],
            "x_label": None,
            "y_label": "Val Label Loss",
            "x_units": "Epoch",
            "y_units": None,
            }, 
    ]
    do_graph(axes, "Source Train Label Loss vs Source Val Label Loss", graphs)
    # Vanilla_Train_Eval_Test_Jig.do_diagram(experiment["history"], axes[0][0])



    if show_only:
        plt.show()
    else:
        plt.savefig(loss_curve_path)


if __name__ == "__main__":
    import sys
    do_report(sys.argv[1], None, show_only=True)