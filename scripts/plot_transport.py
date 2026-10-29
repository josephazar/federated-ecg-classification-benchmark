from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BLACK='#222222'
BLUE='#0072B2'
GREEN='#009E73'
ORANGE='#E69F00'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':BLACK,
 'axes.labelcolor':BLACK,'axes.edgecolor':BLACK,'axes.titlecolor':BLACK,'xtick.color':BLACK,'ytick.color':BLACK,
 'hatch.color':BLACK,'legend.edgecolor':'#cccccc'})
d=pd.read_csv(ROOT/'results/benchmark/integration_summary.csv').set_index(['model','policy'])
fig,ax=plt.subplots(1,2,figsize=(6.5,3.8),layout='constrained')
keys=[('mlp','fixed_L3'),('mlp','adaptive'),('mlp_rr','fixed_L3'),('mlp_rr','adaptive')]
raw=np.array([d.loc[k,'raw_bytes_mean']/1e6 for k in keys]);extra=np.array([d.loc[k,'protection_extra_bytes_mean']/1e6 for k in keys])
ax[0].bar(np.arange(4),raw,color='#C4DCEE',edgecolor=BLUE,label='Tensor data')
ax[0].bar(np.arange(4),extra,bottom=raw,color=ORANGE,edgecolor=BLACK,label='Protection')
ax[0].set(xticks=np.arange(4),xticklabels=['MLP\nL3','MLP\nAdaptive','RR\nL3','RR\nAdaptive'],ylabel='Traffic over 30 rounds (MB)',ylim=(0,245),title='A. Communication')
ax[0].legend(loc='upper left',fontsize=10,frameon=False)
for i,(model,label) in enumerate([('mlp','MLP'),('mlp_rr','MLP+RR')]):
    policies=['none','fixed_L3','adaptive'];xx=np.arange(3)+(i-.5)*.35
    ax[1].bar(xx,[d.loc[(model,p),'elapsed_rounds_seconds_mean'] for p in policies],width=.34,
              yerr=[d.loc[(model,p),'elapsed_rounds_seconds_std'] for p in policies],
              facecolor='white',edgecolor=[BLUE,GREEN][i],hatch='' if i==0 else '///',
              error_kw={'ecolor':BLACK,'capsize':3},label=label)
ax[1].set(xticks=np.arange(3),xticklabels=['None','L3','Adaptive'],ylabel='Sequential host time (s)',ylim=(0,6),title='B. Host execution')
ax[1].legend(loc='upper left',fontsize=10,frameon=False)
for a in ax:
    a.spines[['right','top']].set_visible(False)
    a.tick_params(axis='x',labelsize=10)
out=ROOT/'figures/transport_costs.png'
fig.savefig(out,dpi=260,facecolor='white');plt.close(fig)
print(out)
