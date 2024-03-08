import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib
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
    "legend.fontsize": 7,
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

p_wmmse = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_wmmse_v1_rebuttal.npy')
p_tts = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_tts_v1_rebuttal.npy')
p_zo_cap = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_zo_cap_v1_6GHz_rebuttal.npy')

dfactor = 10
x = np.arange(0, p_wmmse.shape[1], dfactor)


method_list = ['WMMSE (Randomized IRS)',
                'TTS-SSCO: AA',  
                'ZoSGA: Physical IRS (EM)'
]

confidence_interval = 0.95
plt.rcParams.update(tex_fonts)
plt.ylim(3.3,3.8)
plt.xlim(-50, p_wmmse.shape[1])

savgol_window = 500
savgol_polyorder = 4

#blue
avg_p_wmmse = np.mean(p_wmmse,axis=0)
avg_sav_p_wmmse = savgol_filter(avg_p_wmmse, savgol_window, savgol_polyorder)
ci_wmmse = st.t.interval(confidence_interval, len(p_wmmse)-1, loc=np.mean(p_wmmse,axis=0), scale=st.sem(p_wmmse,axis=0))
plt.fill_between(x,np.take(ci_wmmse[0], x), np.take(ci_wmmse[1], x), color='b', alpha=.4)
# plt.plot(x, np.take(avg_p_wmmse, x), color='b',linewidth=1, alpha=.5)

#red
avg_p_tts = np.mean(p_tts,axis=0)
avg_sav_p_tts = savgol_filter(avg_p_tts, savgol_window, savgol_polyorder)
ci_tts = st.t.interval(confidence_interval, len(p_tts)-1, loc=np.mean(p_tts,axis=0), scale=st.sem(p_tts,axis=0))
plt.fill_between(x,np.take(ci_tts[0], x), np.take(ci_tts[1], x), color='r', alpha=.4)
# plt.plot(x, np.take(avg_p_tts, x), color='r', linewidth=1, alpha=.5)

#Black
avg_p_zo_cap = np.mean(p_zo_cap, axis=0)
avg_sav_p_zo_cap = savgol_filter(avg_p_zo_cap, savgol_window, savgol_polyorder)
ci_zo_cap = st.t.interval(confidence_interval, len(p_zo_cap)-1, loc=np.mean(p_zo_cap,axis=0), scale=st.sem(p_zo_cap,axis=0))
plt.fill_between(x,np.take(ci_zo_cap[0], x), np.take(ci_zo_cap[1], x), color='#9370DB', alpha=.4)
# plt.plot(x, np.take(avg_p_zo_cap, x), color='g', linewidth=1, alpha=.3)

plt.plot(x, np.take(avg_sav_p_wmmse, x), color='#1434A4', label=method_list[0], linewidth=1.5)
plt.plot(x, np.take(avg_sav_p_tts, x), color='#811331',label=method_list[1],linewidth=1.5)
plt.plot(x, np.take(avg_sav_p_zo_cap, x), color='#4B0082',label=method_list[2],linewidth=1.5)

print("Done!")
plt.xlabel(r'Iteration (Channel Realization)')
plt.ylabel(r'Sumrate')
# plt.title("$\\beta_{AI}=\\beta_{Iu}=5dB$, $\\beta_{Au}=-5dB$, $r_r = 0.5, r_d = 0, r_{r,k}=(k-1)/3$")
plt.title("$\\beta_{AI}=\\beta_{Iu}=5$dB, $\\beta_{Au}=-5$dB, $r_r = 0.5, r_d = 0, r_{r,k}=(k-1)/3$")
plt.legend(loc='upper left')
plt.tight_layout(pad=0.5)
plt.subplots_adjust(top=0.95)
plt.grid(True)
tikzplotlib.save("../tex/irs1_cap_conv_rebuttal.tex")
plt.savefig('../pdfs/irs1_cap_conv_rebuttal.pdf')
plt.savefig('../irs1_c_conv_rebuttal.png')
plt.show()
# os._exit(0)