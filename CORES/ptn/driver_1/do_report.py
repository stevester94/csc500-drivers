#! /usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.gridspec
import json
from steves_utils.ptn_train_eval_test_jig import PTN_Train_Eval_Test_Jig
import pandas as pds
import matplotlib.patches as mpatches


def do_report(experiment_json_path, loss_curve_path, show_only=False):

    with open(experiment_json_path) as f:
        experiment = json.load(f)

    fig, axes = plt.subplots(2, 2)
    plt.tight_layout()

    fig.suptitle("Experiment Summary")
    fig.set_size_inches(30, 15)

    plt.subplots_adjust(hspace=0.4)
    plt.rcParams['figure.dpi'] = 163

    ###
    # Get Loss Curve
    ###
    PTN_Train_Eval_Test_Jig.do_diagram(experiment["history"], axes[0][0])

    ###
    # Get Results Table
    ###
    ax = axes[0][1]
    ax.set_axis_off() 
    ax.set_title("Results")
    t = ax.table(
        [
            ["Source Val Label Accuracy", "{:.2f}".format(experiment["results"]["source_val_label_accuracy"])],
            ["Source Val Label Loss", "{:.2f}".format(experiment["results"]["source_val_label_loss"])],
            ["Target Val Label Accuracy", "{:.2f}".format(experiment["results"]["target_val_label_accuracy"])],
            ["Target Val Label Loss", "{:.2f}".format(experiment["results"]["target_val_label_loss"])],

            ["Source Test Label Accuracy", "{:.2f}".format(experiment["results"]["source_test_label_accuracy"])],
            ["Source Test Label Loss", "{:.2f}".format(experiment["results"]["source_test_label_loss"])],
            ["Target Test Label Accuracy", "{:.2f}".format(experiment["results"]["target_test_label_accuracy"])],
            ["Target Test Label Loss", "{:.2f}".format(experiment["results"]["target_test_label_loss"])],
            ["Total Epochs Trained", "{:.2f}".format(experiment["results"]["total_epochs_trained"])],
            ["Total Experiment Time Secs", "{:.2f}".format(experiment["results"]["total_experiment_time_secs"])],
        ],
        loc="best",
        cellLoc='left',
        colWidths=[0.3,0.4],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(20)
    t.scale(1.5, 2)


    ###
    # Get Parameters Table
    ###
    ax = axes[1][0]
    ax.set_axis_off() 
    ax.set_title("Parameters")

    t = ax.table(
        [
            ["Experiment Name", experiment["parameters"]["experiment_name"]],
            ["Learning Rate", experiment["parameters"]["lr"]],
            ["Num Epochs", experiment["parameters"]["n_epoch"]],
            ["patience", experiment["parameters"]["patience"]],
            ["seed", experiment["parameters"]["seed"]],
            ["dataset seed", experiment["parameters"]["dataset_seed"]],
            ["device", experiment["parameters"]["device"]],
            ["Source Domains", str(experiment["parameters"]["source_domains"])],
            ["Target Domains", str(experiment["parameters"]["target_domains"])],

            ["Num examples per class per domain", experiment["parameters"]["num_examples_per_class_per_domain"]],
            ["Num shot", experiment["parameters"]["n_shot"]],
            ["Num way", experiment["parameters"]["n_way"]],
            ["Num query", experiment["parameters"]["n_query"]],
            ["Num epoch", experiment["parameters"]["n_epoch"]],
            ["Train k factor", experiment["parameters"]["train_k_factor"]],
            ["Val k factor", experiment["parameters"]["val_k_factor"]],
            ["Test k factor", experiment["parameters"]["test_k_factor"]],

        ],
        loc="best",
        cellLoc='left',
        colWidths=[0.3,0.45],
    )
    t.auto_set_font_size(False)
    t.set_fontsize(20)
    t.scale(1.5, 2)



    #
    # Build a damn pandas dataframe for the per domain accuracies and plot it
    # 

    ax = axes[1][1]
    ax.set_title("Per Domain Validation Accuracy")

    # Convert the dict to a list of tuples
    per_domain_accuracy = experiment["results"]["per_domain_accuracy"]
    per_domain_accuracy = [(domain, v["accuracy"], v["source?"]) for domain,v in per_domain_accuracy.items()]


    df = pds.DataFrame(per_domain_accuracy, columns=["domain", "accuracy", "source?"])
    df.domain = df.domain.astype(float)
    df = df.set_index("domain")
    df = df.sort_values("domain")

    domain_colors = {True: 'r', False: 'b'}
    df['accuracy'].plot(kind='bar', color=[domain_colors[i] for i in df['source?']], ax=ax)

    source_patch = mpatches.Patch(color=domain_colors[True], label='Source Domain')
    target_patch = mpatches.Patch(color=domain_colors[False], label='Target Domain')
    ax.legend(handles=[source_patch, target_patch])
    ax.set_ylim([0.0, 1.0])

    if show_only:
        plt.show()
    else:
        plt.savefig(loss_curve_path)


if __name__ == "__main__":
    import sys
    do_report(sys.argv[1], None, show_only=True)