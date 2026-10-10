"""Leakage-controlled, exploratory decoder improvement after a negative gate pilot."""
import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from data import DEV,load_bank,load_host
from run import COVERAGES,SNR,estimate_features,fit_decoder,metrics,paired_summary,power,score,terms

METHODS=['original_decoder','real_artifact_augmented','estimated_corrected_augmented']


def corrected(p,h,ratio,ref,floor):
    estimated=np.maximum(h-ref,0)*ratio
    return np.maximum(p-estimated,floor)


def run(repo,cache,out,include_switch=False,full_bank=False):
    out.mkdir(parents=True,exist_ok=True);mne.set_log_level('ERROR')
    bank=load_bank(repo,cache,offline_cached=not full_bank)
    donor_fold=np.array([DEV.index(int(s))%3 for s in bank['subject']])
    lam=4.4/(bank['safety_rms'][:,None]*10**(SNR[None,:]/20))
    rows=[];clean_rows=[];pair_rows=[]
    for subject in range(1,10):
        host=load_host(subject,cache);x,h,y,runs=[host[k] for k in ['decoder','high','y','run']]
        for test_run in range(6):
            cal_run=(test_run+1)%6;fit=(runs!=test_run)&(runs!=cal_run)
            csp,original=fit_decoder(x[fit],y[fit]);w=csp.filters_[:8]
            mixed,px,pa=terms(w,x,bank['decoder'],lam)
            mh,phost,ph=terms(w,h,bank['high'],lam)
            reference=np.median(phost[fit],axis=0)
            floor=np.maximum(np.quantile(px[fit],.01,axis=0)*.05,1e-12)
            switch_threshold=float(np.quantile(phost[fit].mean(-1),.95))
            original_clean=score(px,original)>0
            for fold in range(3):
                ta=donor_fold!=fold;ea=~ta;cal=runs==cal_run;test=runs==test_run
                assert not set(bank['subject'][ta])&set(bank['subject'][ea])
                ratio=np.exp(np.median(np.log(np.maximum(pa[ta],1e-12))-
                                      np.log(np.maximum(ph[ta],1e-12)),axis=0))
                corrected_all=corrected(mixed,mh,ratio,reference,floor)
                clean_corrected=corrected(px,phost,ratio,reference,floor)
                fitix=np.ix_(fit,ta,np.ones(len(SNR),bool))
                calix=np.ix_(cal,ta,np.ones(len(SNR),bool))
                testix=np.ix_(test,ea,np.ones(len(SNR),bool))
                fdirty=np.log(mixed[fitix]).reshape(-1,8)
                fcorrected=np.log(corrected_all[fitix]).reshape(-1,8)
                repeats=int(np.sum(ta)*len(SNR))
                yaug=np.repeat(y[fit],repeats)
                # Repeated clean samples implement equal clean/contaminated weight.
                fclean=np.repeat(np.log(px[fit]),repeats,axis=0)
                fclean_corr=np.repeat(np.log(clean_corrected[fit]),repeats,axis=0)
                ytrain=np.r_[yaug,yaug]
                augmented=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(np.r_[fclean,fdirty],ytrain)
                adjusted=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(np.r_[fclean_corr,fcorrected],ytrain)
                ytest=np.broadcast_to(y[test,None,None],mixed[testix].shape[:-1]).ravel()
                iscleanok=np.broadcast_to((original_clean[test]==y[test])[:,None,None],mixed[testix].shape[:-1]).ravel()
                by_snr=np.tile(SNR,int(np.sum(test)*np.sum(ea)))
                paired=dict(subject=np.full(len(ytest),subject),run=np.full(len(ytest),test_run),
                    donor_fold=np.full(len(ytest),fold),snr=by_snr,
                    donor_subject=np.tile(np.repeat(bank['subject'][ea],len(SNR)),int(np.sum(test))),
                    trial_index=np.repeat(np.flatnonzero(test),int(np.sum(ea)*len(SNR))))
                variants=[('original_decoder',original,mixed,px),
                          ('real_artifact_augmented',augmented,mixed,px),
                          ('estimated_corrected_augmented',adjusted,corrected_all,clean_corrected)]
                if include_switch:
                    variants+= [('high95_switch_augmented',augmented,mixed,px),
                                ('high95_switch_corrected',adjusted,corrected_all,clean_corrected)]
                for name,lda,features,clean_features in variants:
                    testscore=score(features[testix],lda).ravel()
                    calscore=score(features[calix],lda).ravel()
                    clean_score_variant=score(clean_features[test],lda)
                    if name.startswith('high95_switch_'):
                        testscore=np.where(mh[testix].mean(-1).ravel()>switch_threshold,
                                           testscore,score(mixed[testix],original).ravel())
                        calscore=np.where(mh[calix].mean(-1).ravel()>switch_threshold,
                                          calscore,score(mixed[calix],original).ravel())
                        clean_score_variant=np.where(phost[test].mean(-1)>switch_threshold,
                                                     clean_score_variant,score(px[test],original))
                    safe=np.abs(testscore);error=(testscore>0)!=ytest;harm=iscleanok&error
                    paired[name+'_error']=error;paired[name+'_harm']=harm;paired[name+'_safe']=safe
                    for scope,mask in [('all',np.ones(len(error),bool))]+[(str(int(s)),by_snr==s) for s in SNR]:
                        for coverage in COVERAGES:
                            rows.append(dict(subject=subject,run=test_run,donor_fold=fold,method=name,
                                scope=scope,coverage=coverage,evaluation='rank',
                                **metrics(safe[mask],error[mask],harm[mask],iscleanok[mask],coverage)))
                    threshold=float(np.quantile(np.abs(calscore),.1))
                    rows.append(dict(subject=subject,run=test_run,donor_fold=fold,method=name,
                        scope='all',coverage=.9,evaluation='calibrated_threshold',threshold=threshold,
                        **metrics(safe,error,harm,iscleanok,.9,threshold)))
                    clean_error=(clean_score_variant>0)!=y[test]
                    clean_rows.append(dict(subject=subject,run=test_run,donor_fold=fold,method=name,
                        n=int(np.sum(test)),errors=int(clean_error.sum()),accuracy=float(np.mean(~clean_error))))
                pair_rows.append(paired)
            print(f'augmentation S{subject} run{test_run} complete',flush=True)
        pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
        pd.DataFrame(clean_rows).to_csv(out/'clean_metrics.csv',index=False)
    df=pd.DataFrame(rows);summary=[]
    for (method,scope,coverage,evaluation),d in df.groupby(['method','scope','coverage','evaluation']):
        summary.append(dict(method=method,scope=scope,coverage=coverage,evaluation=evaluation,
            accepted=int(d.accepted.sum()),n=int(d.n.sum()),error_risk=float(d.error_accepted.sum()/d.accepted.sum()),
            harm_risk=float(d.harm_accepted.sum()/d.accepted.sum()),actual_coverage=float(d.accepted.sum()/d.n.sum()),
            correct_rejected=int(d.correct_rejected.sum())))
    pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
    methods=METHODS+(['high95_switch_augmented','high95_switch_corrected'] if include_switch else [])
    comparisons=[{**paired_summary(df,m,'original_decoder',scope,cov),'coverage':cov}
                 for m in methods[1:]
                 for scope in ['all','-15'] for cov in [.9,1.]]
    result=dict(status='completed_exploratory_followup',n_evaluation_pairs=sum(len(q['subject']) for q in pair_rows),
        methods=methods,comparisons=comparisons,include_switch=include_switch,
        n_donors=len(bank['id']),donor_subjects=np.unique(bank['subject']).tolist(),
        full_bank=full_bank,
        protocol_sha256=hashlib.sha256((Path(__file__).parent/'AUGMENTATION_PROTOCOL.md').read_bytes()).hexdigest(),
        switch_protocol_sha256=hashlib.sha256((Path(__file__).parent/'SWITCHING_PROTOCOL.md').read_bytes()).hexdigest() if include_switch else None,
        limitations=['Follow-up after observing first pilot; not independent confirmation.',
            f'Nine host subjects; {len(bank["id"])} real donors from {len(np.unique(bank["subject"]))} artifact people.',
            'Training on contaminated data is not itself novel.',
            'No online safety, clinical or real-time validation.'])
    (out/'RESULT.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(out/'evaluation_pairs.npz',**{k:np.concatenate([q[k] for q in pair_rows]) for k in pair_rows[0]})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'results_augmentation')
    p.add_argument('--include-switch',action='store_true')
    p.add_argument('--full-bank',action='store_true')
    args=p.parse_args();run(Path(__file__).resolve().parents[2],args.cache,args.out,args.include_switch,args.full_bank)
