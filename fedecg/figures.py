"""Scientific benchmark figures with standard colours and black labels."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .common import ROOT

BLACK='#222222'
BLUE='#0072B2'
ORANGE='#E69F00'
GREEN='#009E73'
PALETTE=[BLUE,ORANGE,GREEN]
OUT=ROOT/'figures'


def setup():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'text.color':BLACK,'axes.labelcolor':BLACK,
        'axes.edgecolor':BLACK,'axes.titlecolor':BLACK,'xtick.color':BLACK,'ytick.color':BLACK,
        'grid.color':'#888888','legend.edgecolor':'#cccccc','axes.spines.top':False,'axes.spines.right':False,
        'savefig.dpi':260})


def save(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    setup();root=ROOT/'results/validation_controls'
    folds=pd.read_csv(root/'grouped_cv_folds.csv');oof=pd.read_csv(root/'grouped_cv_summary.csv')
    recipes=[('mlp','ce','MLP / CE','o','-'),('mlp_rr','ce','MLP + RR / CE','s','--'),('mlp_rr','balanced','MLP + RR / balanced CE','^',':')]
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.15))
    for i,(model,loss,label,marker,style) in enumerate(recipes):
        color=PALETTE[i]
        z=folds[(folds.model==model)&(folds.loss==loss)].groupby('fold').macro_f1.agg(['mean','std'])
        axes[0].errorbar(np.arange(1,4)+.04*(i-1),z['mean'],yerr=z['std'],label=label,color=color,
                         marker=marker,linestyle=style,capsize=3,linewidth=1)
        vals=oof[(oof.model==model)&(oof.loss==loss)].oof_macro_f1
        axes[1].bar(i,vals.mean(),yerr=vals.std(),color='white',edgecolor=color,hatch=['','///','xx'][i],capsize=4,
                     error_kw={'ecolor':color},width=.65)
        for v in vals:axes[1].plot(i,v,marker,ms=3,color=color)
    axes[0].set(xticks=[1,2,3],xlabel='Held-out patient group',ylabel='Validation macro F1',title='Fixed round 30 by patient group',ylim=(.18,.36))
    axes[0].legend(fontsize=6.8,loc='lower left',labelcolor=BLACK)
    axes[1].set(xticks=range(3),xticklabels=['MLP\nCE','MLP + RR\nCE','MLP + RR\nbalanced CE'],ylabel='Pooled out-of-fold macro F1',title='Internal DS1 validation',ylim=(.18,.36))
    for ax in axes:ax.grid(axis='y',alpha=.1)
    save(fig,'patient_group_validation')

    summary=pd.read_csv(ROOT/'results/run_summary.csv');eq=pd.read_csv(root/'equal_steps_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.05))
    sizes=[6396,20000,40142];means=[];sds=[]
    for n in sizes:
        z=summary[summary.run.str.startswith(f'volume_mlp_ce_{n}_')] if n<40142 else summary[(summary.model=='mlp')&(summary.loss=='ce')&summary.run.str.startswith('main_')]
        means.append(z.best_macro_f1.mean());sds.append(z.best_macro_f1.std())
    axes[0].errorbar(np.array(sizes)/1000,means,yerr=sds,color=BLUE,marker='o',capsize=4)
    axes[0].set(xlabel='Available training beats (thousands)',ylabel='Test macro F1',title='Equal epochs: data and steps vary',ylim=(.29,.38))
    g=eq.groupby('train_size').test_final_macro_f1.agg(['mean','std'])
    axes[1].bar(range(2),g['mean'],yerr=g['std'],facecolor='white',edgecolor=[BLUE,GREEN],hatch='///',capsize=4,error_kw={'ecolor':BLACK})
    axes[1].set(xticks=[0,1],xticklabels=['6396','40142'],xlabel='Available unique training beats',ylabel='Fixed final-round test macro F1',
                title='Matched updates and exposures',ylim=(.29,.38))
    for ax in axes:ax.grid(axis='y',alpha=.1)
    save(fig,'training_budget_control')

    z=pd.read_csv(root/'leave_one_patient_out_influence.csv').groupby('omitted_record').difference.agg(['mean','min','max']).sort_index()
    full=pd.read_csv(ROOT/'results/run_summary.csv')
    a=full[full.run.str.startswith('main_')&(full.model=='mlp')&(full.loss=='ce')].best_macro_f1.mean()
    b=full[full.run.str.startswith('main_')&(full.model=='mlp_rr')&(full.loss=='balanced')].best_macro_f1.mean()
    fig,ax=plt.subplots(figsize=(6.6,4.9));y=np.arange(len(z))
    ax.errorbar(z['mean'],y,xerr=np.vstack([z['mean']-z['min'],z['max']-z['mean']]),fmt='o',color=BLUE,elinewidth=.8,capsize=2,ms=3)
    ax.axvline(0,color='#777777',linestyle=':',linewidth=1)
    ax.axvline(b-a,color=BLACK,linestyle='--',linewidth=1,label='All-patient mean effect')
    ax.set(yticks=y,yticklabels=[str(r) for r in z.index],xlabel='Candidate minus baseline macro F1',ylabel='Patient record omitted for diagnostic only',
           title='Sensitivity to each of the 22 test patients')
    ax.invert_yaxis();ax.legend(fontsize=8,labelcolor=BLACK);ax.grid(axis='x',alpha=.1)
    for label in ax.get_yticklabels():
        if label.get_text()=='232':label.set_fontweight('bold')
    save(fig,'patient_influence')


if __name__=='__main__':main()
