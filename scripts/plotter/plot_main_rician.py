import os
from random import shuffle
import numpy as np
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

p_wmmse_b = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/rician/plot_wmmse_b.npy')
p_tts_b = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/rician/plot_tts_b.npy')
p_zo_b = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/rician/plot_zo_b.npy')

method_list = ['WMMSE with Random IRS', 'TTS-SSCO', 'ZOSGA with WMMSE (Proposed)']

sns.set_style("whitegrid")
sns.set_context("paper", rc={"lines.line_width":0.1})
confidence_interval = 0.97
plt.rcParams.update(tex_fonts)
fig, ax = plt.subplots()
plt.ylim(3.45,4.2)

savgol_window = 500
savgol_polyorder = 4

avg_p_wmmse = np.mean(p_wmmse_b,axis=1)
ax.plot(avg_p_wmmse, color='b', marker='o', label=method_list[0],linewidth=1)

avg_p_tts = np.mean(p_tts_b,axis=1)
ax.plot(avg_p_tts, color='r', marker='^', label=method_list[1],linewidth=1)

avg_p_zo = np.mean(p_zo_b, axis=1)
ax.plot(avg_p_zo, color='g', marker='D', label=method_list[2],linewidth=1)

print("Done!")
beta_iu_db = np.array([-6,-4,-2,0,2,4,6,8,10,12,14])
xi = np.arange(beta_iu_db.shape[0])
ax.set_xticks(xi, beta_iu_db)
    #plotting annotations
ax.set_xlabel("Rician Factor $\\beta$(dB)")
ax.set_ylabel("Average Sumrate")
ax.set_title("$\\beta_{AI}=\\beta_{Iu}=\\beta$, $r_r = r_d = r_{r,k}=0$")
ax.legend(loc='upper left')
plt.tight_layout(pad=0.5)
plt.subplots_adjust(top=0.95)
ax.grid(True)
# ax.show()
plt.savefig('../pdfs/main_rician.pdf')
