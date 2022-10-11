import os
from random import shuffle
import numpy as np
from numpy import random
import copy
from collections import Counter
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import seaborn as sns
# from utils.movmean import movmean
from scipy.signal import savgol_filter

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
sns.set(style="whitegrid")
sns.set_context("paper", rc={"lines.line_width":0.1})
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family":"Times New Roman",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.titlesize": 9,
    "axes.labelsize": 10,
    "font.size": 9,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    # reduce padding of x/y label ticks with plots
    "xtick.major.pad":0,
    "ytick.major.pad":0,
    #set figure size and dpi
    'figure.figsize': (4.875, 3.69),
    'figure.dpi': 600
}


p_wmmse = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_wmmse_v2.npy')
p_zo1 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo1_v2.npy')
p_zo2 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo2_v2.npy')
p_zo = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo_v2.npy')



method_list = ['WMMSE Random IRSs', 'Z-SGA with WMMSE IRS-2', 'Z-SGA with WMMSE IRS-1', 'Z-SGA with WMMSE Both IRSs']

confidence_interval = 0.97
plt.rcParams.update(tex_fonts)
plt.ylim(3.3,4.2)
plt.xlim(0, p_wmmse.shape[1])

savgol_window = 500
savgol_polyorder = 4

avg_p_wmmse = np.mean(p_wmmse,axis=0)
avg_sav_p_wmmse = savgol_filter(avg_p_wmmse, savgol_window, savgol_polyorder)
ci_wmmse = (1+confidence_interval) * np.std(p_wmmse, axis=0)/np.sqrt(p_wmmse.shape[0])
plt.fill_between(np.arange(p_wmmse.shape[1]),(avg_p_wmmse-ci_wmmse), (avg_p_wmmse+ci_wmmse), color='b', alpha=.3)
plt.plot(avg_p_wmmse, color='b', linewidth=1)
plt.plot(avg_sav_p_wmmse, color='#1434A4',label=method_list[0],linewidth=1)

avg_p_zo2 = np.mean(p_zo2,axis=0)
avg_sav_p_zo2 = savgol_filter(avg_p_zo2, savgol_window, savgol_polyorder)
ci_zo2 = (1+confidence_interval) * np.std(p_zo2, axis=0)/np.sqrt(p_zo2.shape[0])
plt.fill_between(np.arange(p_zo2.shape[1]),(avg_p_zo2-ci_zo2), (avg_p_zo2+ci_zo2), color='#FFFF00', alpha=.3)
plt.plot(avg_p_zo2, color='#FFFF00', linewidth=1)
plt.plot(avg_sav_p_zo2, color='#F6BE00',label=method_list[1],linewidth=1)

avg_p_zo1 = np.mean(p_zo1,axis=0)
avg_sav_p_zo1 = savgol_filter(avg_p_zo1, savgol_window, savgol_polyorder)
ci_zo1 = (1+confidence_interval) * np.std(p_zo1, axis=0)/np.sqrt(p_zo1.shape[0])
plt.fill_between(np.arange(p_zo1.shape[1]),(avg_p_zo1-ci_zo1), (avg_p_zo1+ci_zo1), color='r', alpha=.3)
plt.plot(avg_p_zo1, color='r', linewidth=1)
plt.plot(avg_sav_p_zo1, color='#811331',label=method_list[2],linewidth=1)

avg_p_zo = np.mean(p_zo, axis=0)
avg_sav_p_zo = savgol_filter(avg_p_zo, savgol_window, savgol_polyorder)
ci_zo = (1+confidence_interval) * np.std(p_zo, axis=0)/np.sqrt(p_zo.shape[0])
plt.fill_between(np.arange(p_zo.shape[1]),(avg_p_zo-ci_zo), (avg_p_zo+ci_zo), color='g', alpha=.3)
plt.plot(avg_p_zo, color='g', linewidth=1)
plt.plot(avg_sav_p_zo, color='#013220',label=method_list[3],linewidth=1)

print("Done!")
plt.xlabel(r'Iteration')
plt.ylabel(r'Sumrate')
plt.title("$\\beta_{AI}=\\beta_{Iu}=5dB$, $r_r = 0.5, r_d = 0, r_{r,k}=(k-1)/3$")
plt.legend(loc='upper left')
plt.tight_layout(pad=0.5)
plt.subplots_adjust(top=0.95)
plt.grid(True)
# plt.show()
plt.savefig('pdfs/main_v2.pdf')
os._exit(00)