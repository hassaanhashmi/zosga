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

p_wmmse = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_wmmse_v1_rebuttal_100.npy')
p_tts = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_tts_v1_rebuttal_100.npy')
p_zo = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_zo_v1_rebuttal_100.npy')
p_zo_c = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_zo_c_v1_rebuttal_100.npy')
# p_zo_cap = np.load('/home/radio/hassaan/dpgzo/scripts_mp/plot_data/main/v1/plot_zo_cap_v1_6GHz.npy')

dfactor = 10
x = np.arange(0, p_wmmse.shape[1], dfactor)


method_list = ['WMMSE (Randomized IRS)',
                'TTS-SSCO: AA',  
                'ZoSGA (Proposed): UA',
                'ZoSGA (Proposed): AA',
                'TTS-SSCO: UA'
]

confidence_interval = 0.95
plt.rcParams.update(tex_fonts)
plt.ylim(3.5,11.0)
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

# avg_p_tts = np.mean(p_tts2,axis=0)
# avg_sav_p_tts = savgol_filter(avg_p_tts, savgol_window, savgol_polyorder)
# ci_tts = st.t.interval(confidence_interval, len(p_tts2)-1, loc=np.mean(p_tts2,axis=0), scale=st.sem(p_tts,axis=0))
# plt.fill_between(x,np.take(ci_tts[0], x), np.take(ci_tts[1], x), alpha=.4)
# # plt.plot(x, np.take(avg_p_tts, x), color='r', linewidth=1, alpha=.3)
# plt.plot(x, np.take(avg_sav_p_tts, x), color='k', label=method_list[4],linewidth=1.5)


#dark orange
avg_p_zo_c = np.mean(p_zo_c, axis=0)
avg_sav_p_zo_c = savgol_filter(avg_p_zo_c, savgol_window, savgol_polyorder)
ci_zo_c = st.t.interval(confidence_interval, len(p_zo_c)-1, loc=np.mean(p_zo_c,axis=0), scale=st.sem(p_zo_c,axis=0))
plt.fill_between(x,np.take(ci_zo_c[0], x), np.take(ci_zo_c[1], x), color='#FBCEB1', alpha=.4)
# plt.plot(x, np.take(avg_p_zo_c, x), color='#FFAC1C', linewidth=1, alpha=.5)

#green
avg_p_zo = np.mean(p_zo, axis=0)
avg_sav_p_zo = savgol_filter(avg_p_zo, savgol_window, savgol_polyorder)
# ci_zo = (1+confidence_interval) * np.std(p_zo, axis=0)/np.sqrt(p_zo.shape[0])
ci_zo = st.t.interval(confidence_interval, len(p_zo)-1, loc=np.mean(p_zo,axis=0), scale=st.sem(p_zo,axis=0))
plt.fill_between(x,np.take(ci_zo[0], x), np.take(ci_zo[1], x), color='g', alpha=.4)
# plt.plot(x, np.take(avg_p_zo, x), color='g', linewidth=1, alpha=.5)

plt.plot(x, np.take(avg_sav_p_wmmse, x), color='#1434A4', label=method_list[0], linewidth=1.5)
plt.plot(x, np.take(avg_sav_p_tts, x), color='#811331',label=method_list[1],linewidth=1.5)
plt.plot(x, np.take(avg_sav_p_zo_c, x), color='#F28C28',label=method_list[2],linewidth=1.5)
plt.plot(x, np.take(avg_sav_p_zo, x), color='#013220',label=method_list[3],linewidth=1.5)

print("Done!")
plt.xlabel(r'Iteration (Channel Realization)')
plt.ylabel(r'Sumrate')
# plt.title("$\\beta_{AI}=\\beta_{Iu}=5dB$, $\\beta_{Au}=-5dB$, $r_r = 0.5, r_d = 0, r_{r,k}=(k-1)/3$")
plt.title("$\\beta_{AI}=\\beta_{Iu}=5$dB, $\\beta_{Au}=-5$dB, $r_r = 0.5, r_d = 0, r_{r,k}=(k-1)/3$")
plt.legend(loc='best')
plt.tight_layout(pad=0.5)
plt.subplots_adjust(top=0.95)
plt.grid(True)
tikzplotlib.save("../tex/irs1_conv_rebuttal_hsnr.tex")
plt.savefig('../pdfs/irs1_conv_rebuttal_100.pdf')
plt.savefig('../irs1_conv_rebuttal_100.png')
plt.show()
# os._exit(0)