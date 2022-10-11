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

p_wmmse = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_wmmse_v1.npy')
p_tts = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_tts_v1.npy')
p_zo = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_zo_v1.npy')
p_zo_c = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_zo_c_v1.npy')
p_zo_q2 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_zo_q2_v1.npy')
p_zo_q3 = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_zo_q3_v1.npy')


method_list = ['WMMSE with Random IRS',
                'TTS-SSCO $Q=\\infty$ $|\\cdot| \\leq 1 $', 
                'Z-SGA $\\& $WMMSE $Q=2$', 
                'Z-SGA $\\& $WMMSE $Q=3$', 
                'Z-SGA $\\& $WMMSE $Q=\\infty$ $|\\cdot| \\leq 1 $',
                'Z-SGA $\\& $WMMSE $Q=\\infty$ $|\\cdot| = 1 $'
]

confidence_interval = 0.97
plt.rcParams.update(tex_fonts)
plt.ylim(3.5,4.0)
plt.xlim(-50, p_wmmse.shape[1])

savgol_window = 500
savgol_polyorder = 4

#blue
avg_p_wmmse = np.mean(p_wmmse,axis=0)
avg_sav_p_wmmse = savgol_filter(avg_p_wmmse, savgol_window, savgol_polyorder)
ci_wmmse = (1+confidence_interval) * np.std(p_wmmse, axis=0)/np.sqrt(p_wmmse.shape[0])
plt.fill_between(np.arange(p_wmmse.shape[1]),(avg_p_wmmse-ci_wmmse), (avg_p_wmmse+ci_wmmse), color='b', alpha=.2)
plt.plot(avg_p_wmmse, color='b',linewidth=1, alpha=.3)
plt.plot(avg_sav_p_wmmse, color='#1434A4', label=method_list[0], linewidth=1.5)

#red
avg_p_tts = np.mean(p_tts,axis=0)
avg_sav_p_tts = savgol_filter(avg_p_tts, savgol_window, savgol_polyorder)
ci_tts = (1+confidence_interval) * np.std(p_tts, axis=0)/np.sqrt(p_tts.shape[0])
plt.fill_between(np.arange(p_tts.shape[1]),(avg_p_tts-ci_tts), (avg_p_tts+ci_tts), color='r', alpha=.2)
plt.plot(avg_p_tts, color='r', linewidth=1, alpha=.3)
plt.plot(avg_sav_p_tts, color='#811331',label=method_list[1],linewidth=1.5)

#light yellow
avg_p_zo_q2 = np.mean(p_zo_q2, axis=0)
avg_sav_p_zo_q2 = savgol_filter(avg_p_zo_q2, savgol_window, savgol_polyorder)
ci_zo = (1+confidence_interval) * np.std(p_zo, axis=0)/np.sqrt(p_zo.shape[0])
plt.fill_between(np.arange(p_zo.shape[1]),(avg_p_zo_q2-ci_zo), (avg_p_zo_q2+ci_zo), color='#FFFF8F', alpha=.2)
plt.plot(avg_p_zo_q2, color='#FFFF00', linewidth=1, alpha=.3)
plt.plot(avg_sav_p_zo_q2, color='#F6BE00',label=method_list[2],linewidth=1.5)


#dark orange
avg_p_zo_q3 = np.mean(p_zo_q3, axis=0)
avg_sav_p_zo_q3 = savgol_filter(avg_p_zo_q3, savgol_window, savgol_polyorder)
ci_zo = (1+confidence_interval) * np.std(p_zo, axis=0)/np.sqrt(p_zo.shape[0])
plt.fill_between(np.arange(p_zo.shape[1]),(avg_p_zo_q3-ci_zo), (avg_p_zo_q3+ci_zo), color='#FBCEB1', alpha=.2)
plt.plot(avg_p_zo_q3, color='#FFAC1C', linewidth=1, alpha=.3)
plt.plot(avg_sav_p_zo_q3, color='#F28C28',label=method_list[3],linewidth=1.5)


#green
avg_p_zo = np.mean(p_zo, axis=0)
avg_sav_p_zo = savgol_filter(avg_p_zo, savgol_window, savgol_polyorder)
ci_zo = (1+confidence_interval) * np.std(p_zo, axis=0)/np.sqrt(p_zo.shape[0])
plt.fill_between(np.arange(p_zo.shape[1]),(avg_p_zo-ci_zo), (avg_p_zo+ci_zo), color='g', alpha=.2)
plt.plot(avg_p_zo, color='g', linewidth=1, alpha=.3)
plt.plot(avg_sav_p_zo, color='#013220',label=method_list[4],linewidth=1.5)


#brown
avg_p_zo_c = np.mean(p_zo_c, axis=0)
avg_sav_p_zo_c = savgol_filter(avg_p_zo_c, savgol_window, savgol_polyorder)
ci_zo = (1+confidence_interval) * np.std(p_zo, axis=0)/np.sqrt(p_zo.shape[0])
plt.fill_between(np.arange(p_zo.shape[1]),(avg_p_zo_c-ci_zo), (avg_p_zo_c+ci_zo), color='#DAA06D', alpha=.2)
plt.plot(avg_p_zo_c, color='#A52A2A', linewidth=1, alpha=.3)
plt.plot(avg_sav_p_zo_c, color='#6E260E',label=method_list[5],linewidth=1.5)

print("Done!")
plt.xlabel(r'Iteration')
plt.ylabel(r'Sumrate')
plt.title("$\\beta_{AI}=\\beta_{Iu}=5dB$, $\\beta_{Au}=-5dB$ $r_r = 0.5, r_d = 0, r_{r,k}=(k-1)/3$")
plt.legend(loc='upper left')
plt.tight_layout(pad=0.5)
plt.subplots_adjust(top=0.95)
plt.grid(True)
plt.savefig('pdfs/main_v1.pdf')
plt.show()
os._exit(0)