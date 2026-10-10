"""Publication-style plot from saved outcomes; no generated/synthetic evidence."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(root):
    stress=pd.read_csv(root/'results_stress/summary.csv')
    choice=pd.read_csv(root/'results_selector/summary.csv')
    clean=pd.read_csv(root/'results_switching/clean_metrics.csv')
    totals=clean.groupby('method')[['n','errors']].sum()
    cchoice=pd.read_csv(root/'results_selector/clean_summary.csv')
    methods=['original_decoder','real_artifact_augmented','high95_switch_augmented','calibrated_decoder_choice']
    names=['Исходный CSP/LDA','Обучение на артефактах','Переключатель high95','Калиброванный выбор']
    colors=['#586575','#E4A36A','#238B75','#946DB1']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(12.8,5.2),gridspec_kw={'width_ratios':[1.55,1]})
    for m,label,color in zip(methods,names,colors):
        d=(choice if m=='calibrated_decoder_choice' else stress)
        d=d[d.method==m].sort_values('snr')
        axs[0].plot(d.snr,d.error_risk*100,'o-',label=label,color=color,lw=2,ms=4)
    axs[0].axvspan(-15,2,color='#edf2f6',zorder=-1)
    axs[0].set(xlabel='SNR по исходному референсу, дБ',ylabel='Ошибочные команды, %',
               title='Сильные помехи: выигрыш; слабые: ограничение',ylim=(20,52),xticks=[-15,-7,0,6,12,20,30])
    axs[0].grid(axis='y',alpha=.2);axs[0].legend(loc='upper right',fontsize=9,frameon=False)
    values=[]
    for m in methods:
        values.append(float(cchoice[cchoice.method==m].accuracy.iloc[0])*100 if m=='calibrated_decoder_choice'
                      else float(1-totals.loc[m,'errors']/totals.loc[m,'n'])*100)
    y=np.arange(len(methods))
    axs[1].scatter(values,y,c=colors,s=90,zorder=3)
    axs[1].axvline(values[0],color=colors[0],linestyle=':',alpha=.5)
    axs[1].set(yticks=y,yticklabels=names,xlim=(65,80),xlabel='Точность, %',title='Без добавленного артефакта')
    axs[1].invert_yaxis();axs[1].grid(axis='x',alpha=.2);axs[1].set_axisbelow(True)
    for i,v in enumerate(values):axs[1].text(v+.45,i,f'{v:.2f}%',va='center',fontsize=10)
    fig.suptitle('RKNP · пилот на 9 EEG-участниках и 6 донорских людях',fontsize=15,fontweight='bold',x=.06,ha='left')
    fig.text(.06,.018,'100% выдача команд · одинаковые контрольные смеси · повторные варианты не независимы · новизна не установлена',fontsize=9,color='#586575')
    fig.tight_layout(rect=(.02,.045,.99,.93),w_pad=2)
    fig.savefig(root/'RESULT_FIGURE.png',dpi=180);fig.savefig(root/'RESULT_FIGURE.pdf')
    plt.close(fig)


if __name__=='__main__':plot(Path(__file__).parent)
