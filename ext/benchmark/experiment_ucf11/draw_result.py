# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import csv


def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        if row[1] == "Step":
            pass
        else:
            y.append(float(row[2]))
            x.append(int(row[1]))
    return x ,y


def reduce_data(data_x, data_y, interval=3):
    assert len(data_x) == len(data_y)

    total_step = len(data_x)

    new_x = []
    new_y = []

    for index in range(0, total_step, interval):
        new_x.append(data_x[index])
        new_y.append(data_y[index])

    return new_x, new_y


if __name__ == '__main__':
    save_path = "./save/mnist_top1_acc.pdf"

    x1, y1 = readcsv(r"./result/run-ucf11_feature_Classifier_40,60,48,48_20210111_221059_233_logs-tag-val_top_1.csv")
    x2, y2 = readcsv(r"./result/run-ucf11_feature_ClassifierBTT_5,5_20201223_234626_233_logs-tag-val_top_1.csv")
    x3, y3 = readcsv(r"./result/run-ucf11_feature_ClassifierCP_400_20201223_234358_233_logs-tag-val_top_1.csv")
    x4, y4 = readcsv(r"./result/run-ucf11_feature_ClassifierTK2_10,10_20201223_234120_233_logs-tag-val_top_1.csv")
    x5, y5 = readcsv(r"./result/run-ucf11_feature_ClassifierTR_20,20,20,20_20201223_233624_233_logs-tag-val_top_1.csv")
    x6, y6 = readcsv(r"./result/run-ucf11_feature_ClassifierTT_10_20201223_233642_233_logs-tag-val_top_1.csv")

    x1, y1 = reduce_data(x1, y1)
    x2, y2 = reduce_data(x2, y2)
    x3, y3 = reduce_data(x3, y3)
    x4, y4 = reduce_data(x4, y4)
    x5, y5 = reduce_data(x5, y5)
    x6, y6 = reduce_data(x6, y6)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    ax.set_xlim((3, 301))
    ax.set_ylim((0.50, 0.98))

    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_text("Epoch", fontsize=18)
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_label_text("Top-1 Accuracy", fontsize=18)

    color_scheme = dict(ori="#5A7C3C", btt="#65AACB", cp="#4D73BE", tk2="#94AAD8", tr="#DF8244", tt="#68389A")

    f1 = ax.plot(x1, y1, "--", color=color_scheme["ori"], label="LSTM:0.8703", linewidth=2)
    f2 = ax.plot(x2, y2, "-", color=color_scheme["btt"], label="BTT-LSTM:0.8892", linewidth=2)
    f3 = ax.plot(x3, y3, "-", color=color_scheme["cp"], label="CP-LSTM:0.8892", linewidth=2)
    f4 = ax.plot(x4, y4, "-", color=color_scheme["tk2"], label="TK2-LSTM:0.75", linewidth=2)
    f5 = ax.plot(x5, y5, "-", color=color_scheme["tr"], label="TR-LSTM:0.9209", linewidth=2)
    f6 = ax.plot(x6, y6, "-", color=color_scheme["tt"], label="TT-LSTM:0.9019", linewidth=2)

    ax.legend(
              bbox_to_anchor=[0.95, 0.38],
              # labelspacing =0.1,
              fontsize=12,
              # loc="best",
              # loc="lower right",
              )
    # ax.grid()

    plt.tight_layout()
    plt.savefig(save_path)

    save_path = "./save/mnist_cr.pdf"

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    # ax.xaxis.set_ticks_position("bottom")
    # ax.xaxis.set_label_text("Epoch", fontsize=18)
    ax.yaxis.set_ticks_position("left")
    ax.yaxis.set_label_text("CR", fontsize=18)

    data_dict = {"BTT": 164, "CP": 146, "TK2": 177, "TR": 146, "TT": 164}
    model_name = []
    y = []
    for k, v in data_dict.items():
        model_name.append(k)
        y.append(v)

    x = list(range(len(model_name)))

    ax.bar(x, y, color=[color_scheme["btt"], color_scheme["cp"], color_scheme["tk2"], color_scheme["tr"],
                        color_scheme["tt"]], tick_label=model_name)
    for a, b in zip(x, y):
        plt.text(a, b+6, r"%d$\times$" % b, ha='center', va='center', fontsize=13)

    # ax.set_xlim((8, 301))
    ax.set_ylim((0, 250))
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=10)
    # ax.legend(
    #     # bbox_to_anchor=[0.95, 0.85],
    #     # labelspacing =0.1,
    #     # fontsize=10,
    #     # loc="best",
    #     loc="lower right",
    # )
    # ax.grid()

    plt.tight_layout()
    plt.savefig(save_path)

    plt.show()
