"""Challenge the frozen original/augmented switch at weaker amplitudes."""
import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from data import DEV, load_bank, load_host
from run import SNR, fit_decoder, score, terms

EVAL_SNR=np.r_[SNR,[6.,12.,20.,30.]]
METHODS=['original_decoder','real_artifact_augmented','high95_switch_augmented']


def run(repo,cache,out,reference_root=None,full_bank=False):
    out.mkdir(parents=True,exist_ok=True);mne.set_log_level('ERROR')
    bank=load_bank(repo,cache,offline_cached=not full_bank)
    donor_fold=np.array([DEV.index(int(s))%3 for s in bank['subject']])
    train_lam=4.4/(bank['safety_rms'][:,None]*10**(SNR[None,:]/20))
    eval_lam=4.4/(bank['safety_rms'][:,None]*10**(EVAL_SNR[None,:]/20))
    rows=[];clean_rows=[];old_errors={m:[] for m in METHODS}
    for subject in range(1,10):
        host=load_host(subject,cache)
        x,h,y,runs=[host[k] for k in ['decoder','high','y','run']]
        for test_run in range(6):
            cal_run=(test_run+1)%6
            fit=(runs!=test_run)&(runs!=cal_run);test=runs==test_run
            csp,original=fit_decoder(x[fit],y[fit]);w=csp.filters_[:8]
            mixed,px,_=terms(w,x,bank['decoder'],train_lam)
            evaluated,_,_=terms(w,x[test],bank['decoder'],eval_lam)
            mh,phost,_=terms(w,h,bank['high'],eval_lam)
            threshold=float(np.quantile(phost[fit].mean(-1),.95))
            clean_score=score(px[test],original)
            for fold in range(3):
                ta=donor_fold!=fold;ea=~ta
                assert not set(bank['subject'][ta])&set(bank['subject'][ea])
                repeats=int(np.sum(ta)*len(SNR))
                fdirty=np.log(mixed[np.ix_(fit,ta,np.ones(len(SNR),bool))]).reshape(-1,8)
                fclean=np.repeat(np.log(px[fit]),repeats,axis=0)
                ya=np.repeat(y[fit],repeats)
                augmented=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(
                    np.r_[fclean,fdirty],np.r_[ya,ya])
                dirty_power=evaluated[:,ea]
                orig_score=score(dirty_power,original)
                aug_score=score(dirty_power,augmented)
                activate=mh[test][:,ea].mean(-1)>threshold
                switch_score=np.where(activate,aug_score,orig_score)
                expected=np.broadcast_to(y[test,None,None],orig_score.shape)
                clean_ok=np.broadcast_to(((clean_score>0)==y[test])[:,None,None],orig_score.shape)
                variant_clean=score(px[test],augmented)
                clean_activate=phost[test].mean(-1)>threshold
                clean_scores=[clean_score,variant_clean,np.where(clean_activate,variant_clean,clean_score)]
                for method,ss,cs in zip(METHODS,[orig_score,aug_score,switch_score],clean_scores):
                    error=(ss>0)!=expected;harm=clean_ok&error
                    old_errors[method].append(error[:,:,:len(SNR)].ravel())
                    for j,snr in enumerate(EVAL_SNR):
                        rows.append(dict(subject=subject,run=test_run,donor_fold=fold,method=method,
                            snr=float(snr),n=error[:,:,j].size,errors=int(error[:,:,j].sum()),
                            harms=int(harm[:,:,j].sum()),switch_active=int(activate[:,:,j].sum()),
                            switch_threshold=threshold))
                    clean_rows.append(dict(subject=subject,run=test_run,donor_fold=fold,method=method,
                        n=len(cs),errors=int(((cs>0)!=y[test]).sum()),
                        switch_active=int(clean_activate.sum()),switch_threshold=threshold))
            print(f'amplitude challenge S{subject} run{test_run} complete',flush=True)
    df=pd.DataFrame(rows);clean=pd.DataFrame(clean_rows)
    df.to_csv(out/'fold_metrics.csv',index=False);clean.to_csv(out/'clean_metrics.csv',index=False)
    summary=df.groupby(['method','snr'],as_index=False)[['n','errors','harms','switch_active']].sum()
    summary['error_risk']=summary.errors/summary.n
    summary['harm_risk']=summary.harms/summary.n
    summary['switch_fraction']=summary.switch_active/summary.n
    summary.to_csv(out/'summary.csv',index=False)
    clean_summary=clean.groupby('method',as_index=False)[['n','errors','switch_active']].sum()
    clean_summary['accuracy']=1-clean_summary.errors/clean_summary.n
    clean_summary['switch_fraction']=clean_summary.switch_active/clean_summary.n
    clean_summary.to_csv(out/'clean_summary.csv',index=False)
    identical=None
    if reference_root is not None:
        reference=dict(np.load(Path(reference_root)/'evaluation_pairs.npz'))
        for method in METHODS:
            assert np.array_equal(np.concatenate(old_errors[method]),reference[method+'_error'])
        identical=True
    result=dict(status='completed_frozen_recipe_challenge',train_snr=SNR.tolist(),
        evaluated_snr=EVAL_SNR.tolist(),original_grid_error_vectors_identical=identical,
        n_variant_pairs=int(summary[summary.method==METHODS[0]].n.sum()),
        host_people=9,artifact_people=int(len(np.unique(bank['subject']))),
        protocol_sha256=hashlib.sha256((Path(__file__).parent/'STRESS_PROTOCOL.md').read_bytes()).hexdigest(),
        note='Same source trials and people; only amplitude extended. No independent confirmation.')
    (out/'RESULT.json').write_text(json.dumps(result,indent=2))
    print(summary.to_string(index=False),flush=True)
    print(clean_summary.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'results_stress')
    p.add_argument('--reference-root',type=Path,default=Path(__file__).parent/'results_switching')
    p.add_argument('--full-bank',action='store_true');args=p.parse_args()
    run(Path(__file__).resolve().parents[2],args.cache,args.out,args.reference_root,args.full_bank)
