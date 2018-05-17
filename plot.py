import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import argparse
import numpy as np

def plot(dict, figname="./checkpoints/results.png"):
    fig, (train_ax, val_ax) = plt.subplots(nrows=2, ncols=3, figsize=(20,10))

    plot_sub(train_ax, stats["train_loss"], "Train")
    plot_sub(val_ax, stats["val_loss"], "Val")

    fig.tight_layout()
    plt.savefig(figname)

def smooth(y, box_pts = 20):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_sub(train_ax, losses, mode="Train"):
    ax0, ax1, ax2 = train_ax
    length = len(losses["G"])
    if mode == "Train":
        ite = np.arange(length) / 548.0
    else:
        ite = np.arange(length) / 100.0

    ax0.plot(ite[0::10],smooth(losses["G"])[0::10], label="G")
    ax0.plot(ite[0::10],smooth(losses["G_A"])[0::10], label="G_A")
    # ax0.plot(losses["G_A_idt"], label="G_A_idt")
    ax0.plot(ite[0::10],smooth(losses["G_B"])[0::10], label="G_B")
    # ax0.plot(losses["G_B_idt"], label="G_B_idt")
    ax0.set_title("%s G Loss" % mode, fontsize=16, fontweight='bold')
    ax0.set_xlabel("Epoch", fontsize=16, fontweight='bold')
    ax0.set_ylabel("Loss", fontsize=16, fontweight='bold')
    ax0.set_ylim([-1, 7])
    #    ax0.set_xticklabels(np.arange(-20, 101, 20))
    ax0.tick_params(axis='both', which='major', labelsize=16)
    ax0.legend(fontsize=16)

    ax1.plot(ite[0::10],smooth(losses["D"])[0::10], label="D")
    ax1.plot(ite[0::10],smooth(losses["D_A"])[0::10], label="D_A")
    ax1.plot(ite[0::10],smooth(losses["D_B"])[0::10], label="D_B")
    ax1.set_title("%s D Loss" % mode, fontsize=16, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Loss", fontsize=16, fontweight='bold')
    ax1.set_ylim([-0.2, 1.0])
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend(fontsize=16)

    ax2.plot(ite[0::10],smooth(losses["Cyc_A"])[0::10], label="Cyc_A")
    ax2.plot(ite[0::10],smooth(losses["Cyc_B"])[0::10], label="Cyc_B")
    ax2.set_title("%s Cycle Loss" % mode, fontsize=16, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=16, fontweight='bold')
    ax2.set_ylabel("Loss", fontsize=16, fontweight='bold')
    ax2.set_ylim([-1, 3])
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.legend(fontsize=16)

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default="", type=str)
args = parser.parse_args()

# fname = os.path.join("checkpoints", "train.json")
# outname = os.path.join("checkpoints", "result.png")
fname = os.path.join(args.dir, "train.json")
outname = os.path.join(args.dir, "result.png")
with open(fname) as f:
    stats = json.load(f)
    plot(stats, outname)
