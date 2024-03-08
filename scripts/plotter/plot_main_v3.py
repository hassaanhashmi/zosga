import os
import sys
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
import scipy.stats as st
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

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

p_wmmse = np.vstack((p_wmmse,p_wmmse2,p_wmmse3,p_wmmse4))
p_zo = np.vstack((p_zo,p_zo2,p_zo3,p_zo4))
print(p_zo.shape)

np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v3/plot_wmmse_v3_6ghz_50k.npy',p_wmmse)
np.save('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v3/plot_zo_v3_6ghz_50k.npy',p_zo)
dfactor = 10
x = np.arange(0, p_wmmse.shape[1], dfactor)





# method_list = ['WMMSE Random IRSs', 'Z-SGA with WMMSE IRS-2', 'Z-SGA with WMMSE IRS-1', 'Z-SGA with WMMSE Both IRSs']

method_list = ['WMMSE (Randomized IRSs)',
                'ZoSGA: Physical IRSs (EM)'
                ]

confidence_interval = 0.95
plt.rcParams.update(tex_fonts)
plt.ylim(2.7,3.9)
plt.xlim(0, p_wmmse.shape[1])

savgol_window = 500
savgol_polyorder = 4

avg_p_wmmse = np.mean(p_wmmse,axis=0)
avg_sav_p_wmmse = savgol_filter(avg_p_wmmse, savgol_window, savgol_polyorder)
ci_wmmse = st.t.interval(confidence_interval, p_wmmse.shape[0]-1, loc=np.mean(p_wmmse, axis=0), scale=st.sem(p_wmmse, axis=0))
plt.fill_between(x=x,y1=np.take(ci_wmmse[0],x), y2=np.take(ci_wmmse[1],x), color='b', alpha=.3)
# plt.plot(x, np.take(avg_p_wmmse, x), color='b', linewidth=0.5, alpha=.5)
plt.plot(x, np.take(avg_sav_p_wmmse,x), color='#1434A4',label=method_list[0],linewidth=1.5)

avg_p_zo = np.mean(p_zo, axis=0)
avg_sav_p_zo = savgol_filter(avg_p_zo, savgol_window, savgol_polyorder)
ci_zo = st.t.interval(confidence_interval, p_zo.shape[0]-1, loc=np.mean(p_zo, axis=0), scale=st.sem(p_zo, axis=0))
plt.fill_between(x=x,y1=np.take(ci_zo[0], x), y2=np.take(ci_zo[1], x), color='#9370DB', alpha=.3)
# plt.plot(x, np.take(avg_p_zo, x), color='#9370DB', linewidth=0.5, alpha=.5)
plt.plot(x, np.take(avg_sav_p_zo, x), color='#4B0082',label=method_list[1],linewidth=1.5)

print("Done!")
plt.xlabel(r'Iteration (Channel Realization)')
plt.ylabel(r'Sumrate')
plt.title("$\\beta_{AI}=\\beta_{Iu}=5$dB, $\\beta_{Au}=-5$dB, $r_r = 0.5, r_d = 0, r_{r,k}=(k-1)/7$")
plt.legend(loc='upper left')
plt.tight_layout(pad=0.5)
plt.subplots_adjust(top=0.95)
plt.grid(True)
# plt.show()
# tikzplotlib.save("../tex/irs3_conv_rebuttal.tex")
plt.savefig('../pdfs/irs3_conv_95.pdf')
plt.savefig('irs3_conv_95.png')

# os._exit(00)