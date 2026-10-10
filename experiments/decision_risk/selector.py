"""Learn decoder choice on a disjoint calibration run and donor-person folds."""
import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import DEV,load_bank,load_host
from run import SEED,SNR,fit_decoder,score,terms
from stress import EVAL_SNR

METHODS=['original_decoder','real_artifact_augmented','calibrated_decoder_choice']


def choice_features(orig,aug,high,reference):
    return np.stack([np.abs(orig),np.abs(aug),
        np.log(np.maximum(high,1e-12)/max(reference,1e-12))],axis=-1)


def select(orig,aug,features,gate):
    probability=gate.predict_proba(features.reshape(-1,3))[:,1].reshape(orig.shape) if hasattr(gate,'predict_proba') else np.full(orig.shape,gate)
    use=((orig>0)!=(aug>0))&(probability>.5)
    return np.where(use,aug,orig),use


def run(repo,cache,out,reference_root=None,full_bank=False):
    out.mkdir(parents=True,exist_ok=True);mne.set_log_level('ERROR')
    bank=load_bank(repo,cache,offline_cached=not full_bank)
    folds=np.array([DEV.index(int(s))%3 for s in bank['subject']])
    train_lam=4.4/(bank['safety_rms'][:,None]*10**(SNR[None,:]/20))
    eval_lam=4.4/(bank['safety_rms'][:,None]*10**(EVAL_SNR[None,:]/20))
    rows=[];clean_rows=[];pairs=[];old_errors={m:[] for m in METHODS[:2]};fallbacks=0
    for subject in range(1,10):
        host=load_host(subject,cache);x,h,y,runs=[host[k] for k in ['decoder','high','y','run']]
        for test_run in range(6):
            cal_run=(test_run+1)%6;fit=(runs!=test_run)&(runs!=cal_run);cal=runs==cal_run;test=runs==test_run
            assert not np.any(fit&cal) and not np.any(fit&test) and not np.any(cal&test)
            csp,original=fit_decoder(x[fit],y[fit]);w=csp.filters_[:8]
            mixed,px,_=terms(w,x,bank['decoder'],train_lam)
            evaluated,_,_=terms(w,x,bank['decoder'],eval_lam)
            mh,phost,_=terms(w,h,bank['high'],eval_lam)
            reference=float(np.median(phost[fit].mean(-1)))
            clean_orig=score(px,original)
            for fold in range(3):
                ta=folds!=fold;ea=~ta
                assert not set(bank['subject'][ta])&set(bank['subject'][ea])
                repeats=int(np.sum(ta)*len(SNR))
                fdirty=np.log(mixed[np.ix_(fit,ta,np.ones(len(SNR),bool))]).reshape(-1,8)
                fclean=np.repeat(np.log(px[fit]),repeats,axis=0);ya=np.repeat(y[fit],repeats)
                augmented=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(np.r_[fclean,fdirty],np.r_[ya,ya])
                clean_aug=score(px,augmented)
                calix=np.ix_(cal,ta,np.ones(len(EVAL_SNR),bool))
                testix=np.ix_(test,ea,np.ones(len(EVAL_SNR),bool))
                oc=score(evaluated[calix],original).ravel();ac=score(evaluated[calix],augmented).ravel()
                hc=mh[calix].mean(-1).ravel();yc=np.broadcast_to(y[cal,None,None],evaluated[calix].shape[:-1]).ravel()
                # Equal aggregate clean/contaminated calibration weight before disagreement filtering.
                cal_repeats=int(np.sum(ta)*len(EVAL_SNR))
                oc=np.r_[oc,np.repeat(clean_orig[cal],cal_repeats)]
                ac=np.r_[ac,np.repeat(clean_aug[cal],cal_repeats)]
                hc=np.r_[hc,np.repeat(phost[cal].mean(-1),cal_repeats)]
                yc=np.r_[yc,np.repeat(y[cal],cal_repeats)]
                disagreement=(oc>0)!=(ac>0);target=((ac>0)==yc)[disagreement].astype(int)
                if len(np.unique(target))<2:
                    gate=float(target.mean()) if len(target) else 0.;fallbacks+=1
                else:
                    gate=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=SEED))
                    gate.fit(choice_features(oc,ac,hc,reference)[disagreement],target)
                ot=score(evaluated[testix],original);at=score(evaluated[testix],augmented)
                chosen,use=select(ot,at,choice_features(ot,at,mh[testix].mean(-1),reference),gate)
                cc,cu=select(clean_orig[test],clean_aug[test],
                    choice_features(clean_orig[test],clean_aug[test],phost[test].mean(-1),reference),gate)
                expected=np.broadcast_to(y[test,None,None],ot.shape)
                clean_ok=np.broadcast_to(((clean_orig[test]>0)==y[test])[:,None,None],ot.shape)
                record=dict(subject=np.full(ot.size,subject),run=np.full(ot.size,test_run),
                    donor_fold=np.full(ot.size,fold),snr=np.tile(EVAL_SNR,int(np.sum(test)*np.sum(ea))),
                    donor_subject=np.tile(np.repeat(bank['subject'][ea],len(EVAL_SNR)),int(np.sum(test))),
                    trial_index=np.repeat(np.flatnonzero(test),int(np.sum(ea)*len(EVAL_SNR))))
                for method,ss,cs in zip(METHODS,[ot,at,chosen],[clean_orig[test],clean_aug[test],cc]):
                    error=(ss>0)!=expected;harm=clean_ok&error
                    record[method+'_error']=error.ravel();record[method+'_harm']=harm.ravel()
                    if method in old_errors:old_errors[method].append(error[:,:,:len(SNR)].ravel())
                    for j,snr in enumerate(EVAL_SNR):
                        rows.append(dict(subject=subject,run=test_run,donor_fold=fold,method=method,
                            snr=float(snr),n=error[:,:,j].size,errors=int(error[:,:,j].sum()),
                            harms=int(harm[:,:,j].sum()),choice_augmented=int(use[:,:,j].sum())))
                    clean_rows.append(dict(subject=subject,run=test_run,donor_fold=fold,method=method,
                        n=len(cs),errors=int(((cs>0)!=y[test]).sum()),choice_augmented=int(cu.sum())))
                pairs.append(record)
            print(f'calibrated choice S{subject} run{test_run} complete',flush=True)
    df=pd.DataFrame(rows);clean=pd.DataFrame(clean_rows)
    df.to_csv(out/'fold_metrics.csv',index=False);clean.to_csv(out/'clean_metrics.csv',index=False)
    summary=df.groupby(['method','snr'],as_index=False)[['n','errors','harms','choice_augmented']].sum()
    summary['error_risk']=summary.errors/summary.n;summary['harm_risk']=summary.harms/summary.n
    summary['choice_fraction']=summary.choice_augmented/summary.n
    summary.to_csv(out/'summary.csv',index=False)
    clean_summary=clean.groupby('method',as_index=False)[['n','errors','choice_augmented']].sum()
    clean_summary['accuracy']=1-clean_summary.errors/clean_summary.n
    clean_summary.to_csv(out/'clean_summary.csv',index=False)
    np.savez_compressed(out/'evaluation_pairs.npz',**{k:np.concatenate([q[k] for q in pairs]) for k in pairs[0]})
    identical=None
    if reference_root is not None:
        reference_data=dict(np.load(Path(reference_root)/'evaluation_pairs.npz'))
        for method in old_errors:assert np.array_equal(np.concatenate(old_errors[method]),reference_data[method+'_error'])
        identical=True
    result=dict(status='completed_exploratory_calibrated_choice',original_grid_control_errors_identical=identical,
        n_variant_pairs=int(summary[summary.method==METHODS[0]].n.sum()),calibration_constant_fallbacks=fallbacks,
        host_people=9,artifact_people=int(len(np.unique(bank['subject']))),
        protocol_sha256=hashlib.sha256((Path(__file__).parent/'SELECTOR_PROTOCOL.md').read_bytes()).hexdigest(),
        note='Recipe specified after prior outcomes on same public sources; no independent confirmation.')
    (out/'RESULT.json').write_text(json.dumps(result,indent=2))
    print(summary.to_string(index=False),flush=True);print(clean_summary.to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'results_selector')
    p.add_argument('--reference-root',type=Path,default=Path(__file__).parent/'results_switching')
    p.add_argument('--full-bank',action='store_true');args=p.parse_args()
    run(Path(__file__).resolve().parents[2],args.cache,args.out,args.reference_root,args.full_bank)
