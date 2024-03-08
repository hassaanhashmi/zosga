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
import tikzplotlib
# from utils.movmean import movmean
import scipy.stats as st
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
    'figure.dpi': 600,
    'pdf.fonttype' : 42
}


p_wmmse = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_wmmse_v2_c_100k.npy')
p_zo1 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo1_v2_c_100k.npy')
p_zo2 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo2_v2_c_100k.npy')
p_zo = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v2/plot_zo_v2_c_100k.npy')

dfactor = 20
x = np.arange(0, p_wmmse.shape[1], dfactor)

# method_list = ['WMMSE Random IRSs', 'Z-SGA with WMMSE IRS-2', 'Z-SGA with WMMSE IRS-1', 'Z-SGA with WMMSE Both IRSs']

method_list = ['WMMSE (Randomized IRSs)',
                'ZoSGA: UA, IRS 2',
                'ZoSGA: UA, IRS 1',
                'ZoSGA: UA, IRS 1+2'
                ]

confidence_interval = 0.95
plt.rcParams.update(tex_fonts)
plt.ylim(3.3,4.25)
plt.xlim(0, p_wmmse.shape[1])

savgol_window = 500
savgol_polyorder = 4

avg_p_wmmse = np.mean(p_wmmse,axis=0)
avg_sav_p_wmmse = savgol_filter(avg_p_wmmse, savgol_window, savgol_polyorder)
ci_wmmse = st.t.interval(confidence_interval, p_wmmse.shape[0]-1, loc=np.mean(p_wmmse, axis=0), scale=st.sem(p_wmmse, axis=0))
plt.fill_between(x, np.take(ci_wmmse[0], x), np.take(ci_wmmse[1], x), color='b', alpha=.4)
# plt.plot(x, np.take(avg_p_wmmse, x), color='b', linewidth=1, alpha=.3)
plt.plot(x, np.take(avg_sav_p_wmmse, x), color='#1434A4',label=method_list[0],linewidth=1.5)

avg_p_zo2 = np.mean(p_zo2,axis=0)
avg_sav_p_zo2 = savgol_filter(avg_p_zo2, savgol_window, savgol_polyorder)
ci_zo2 = st.t.interval(confidence_interval, p_zo2.shape[0]-1, loc=np.mean(p_zo2, axis=0), scale=st.sem(p_zo2, axis=0))
plt.fill_between(x, np.take(ci_zo2[0], x), np.take(ci_zo2[1], x), color='c', alpha=.4)
# plt.plot(x, np.take(avg_p_zo2, x), color='c', linewidth=1, alpha=.3)
plt.plot(x, np.take(avg_sav_p_zo2, x), color='#0aa6a6',label=method_list[1],linewidth=1.5)

avg_p_zo1 = np.mean(p_zo1,axis=0)
avg_sav_p_zo1 = savgol_filter(avg_p_zo1, savgol_window, savgol_polyorder)
ci_zo1 = st.t.interval(confidence_interval, p_zo1.shape[0]-1, loc=np.mean(p_zo1, axis=0), scale=st.sem(p_zo1, axis=0))
plt.fill_between(x, np.take(ci_zo1[0], x), np.take(ci_zo1[1], x), color='#FFA500', alpha=.4)
# plt.plot(x, np.take(avg_p_zo1, x), color='r', linewidth=1, alpha=.3)
plt.plot(x, np.take(avg_sav_p_zo1, x), color='#de7409',label=method_list[2],linewidth=1.5)

avg_p_zo = np.mean(p_zo, axis=0)
avg_sav_p_zo = savgol_filter(avg_p_zo, savgol_window, savgol_polyorder)
ci_zo = st.t.interval(confidence_interval, p_zo.shape[0]-1, loc=np.mean(p_zo, axis=0), scale=st.sem(p_zo, axis=0))
plt.fill_between(x, np.take(ci_zo[0], x), np.take(ci_zo[1], x), color='g', alpha=.4)
# plt.plot(x, np.take(avg_p_zo, x), color='g', linewidth=1, alpha=.3)
plt.plot(x, np.take(avg_sav_p_zo, x), color='#013220',label=method_list[3],linewidth=1.5)

print("Done!")
plt.xlabel(r'Iteration (Channel Realization)')
plt.ylabel(r'Sumrate')
plt.title("$\\beta_{AI}=\\beta_{Iu}=5$dB, $\\beta_{Au}=-5$dB, $r_r = 0.5, r_d = 0, r_{r,k}=(k-1)/3$")
plt.legend(loc='upper left')
plt.tight_layout(pad=0.5)
plt.subplots_adjust(top=0.95)
plt.grid(True)
# plt.show()
tikzplotlib.save("../tex/irs2_conv.tex")
plt.savefig('../pdfs/irs2_conv.pdf')
plt.savefig('../irs2_conv.png')

# os._exit(00)