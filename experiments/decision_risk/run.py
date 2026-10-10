"""Run the frozen exploratory candidates; labels never enter evaluation features."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from scipy.stats import spearmanr, wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import DEV, load_bank, load_host

SEED=20261010
SNR=np.array([2.,0.,-3.,-7.,-15.])
COVERAGES=[.5,.7,.9,1.]
LEDGER={1:.840278,2:.562500,3:.951389,4:.756944,5:.548611,
        6:.715278,7:.715278,8:.986111,9:.784722}
METHODS=['margin','high_power','lr_confidence','lr_confidence_high',
         'estimated_power_risk','oracle_power_risk']


def power(projections):return np.mean(projections**2,axis=-1)


def score(p,lda):
    return np.log(np.maximum(p,1e-300))@lda.coef_[0]+lda.intercept_[0]


def fit_decoder(x,y):
    csp=CSP(n_components=8,reg=None,log=True,norm_trace=False,
            transform_into='average_power')
    f=csp.fit_transform(x,y)
    lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto').fit(f,y)
    return csp,lda


def terms(w,host,donor,lam):
    z=np.einsum('kc,nct->nkt',w,host,optimize=True)
    u=np.einsum('kc,dct->dkt',w,donor,optimize=True)
    px,pa=power(z),power(u)
    cross=np.stack([z[:,k]@u[:,k].T/z.shape[-1] for k in range(w.shape[0])],axis=-1)
    mixed=px[:,None,None,:]+lam[None,:,:,None]**2*pa[None,:,None,:]
    mixed=mixed+2*lam[None,:,:,None]*cross[:,:,None,:]
    assert np.isfinite(mixed).all() and np.min(mixed)>0
    return mixed,px,pa


def estimate_features(p_dirty,p_high,lda,ratio,reference_high,floor):
    """Inputs are observable dirty powers and training-only fitted parameters."""
    raw=score(p_dirty,lda)
    total_h=np.log(np.maximum(p_high.mean(-1),1e-12))
    estimate=np.maximum(p_high-reference_high,0)*ratio
    corrected=np.maximum(p_dirty-estimate,floor)
    corrected_score=score(corrected,lda)
    basic=np.stack([np.abs(raw),total_h],axis=-1)
    extra=np.stack([np.abs(corrected_score-raw),
                    ((corrected_score>0)!=(raw>0)).astype(float),
                    np.log1p(np.sum(estimate,-1)/np.maximum(np.sum(p_dirty,-1),1e-12)),
                    np.mean(p_dirty-estimate<floor,-1)],axis=-1)
    return np.concatenate([basic,extra],axis=-1),raw,estimate


def oracle_features(p_dirty,p_high,lda,true_art,floor):
    raw=score(p_dirty,lda);corrected=score(np.maximum(p_dirty-true_art,floor),lda)
    basic=np.stack([np.abs(raw),np.log(np.maximum(p_high.mean(-1),1e-12))],-1)
    extra=np.stack([np.abs(corrected-raw),((corrected>0)!=(raw>0)).astype(float),
        np.log1p(np.sum(true_art,-1)/np.maximum(np.sum(p_dirty,-1),1e-12)),
        np.mean(p_dirty-true_art<floor,-1)],-1)
    return np.concatenate([basic,extra],-1)


def fit_risk(features,error):
    if len(np.unique(error))!=2:raise ValueError('Calibration has one outcome class')
    return make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,
                        random_state=SEED)).fit(features,error)


def metrics(safe,error,harm,clean_correct,coverage,threshold=None):
    n=len(error)
    if threshold is None:
        k=int(np.ceil(n*coverage))
        ix=np.argsort(-safe,kind='stable')[:k]
        accepted=np.zeros(n,bool);accepted[ix]=True
    else:accepted=safe>=threshold
    k=int(accepted.sum())
    return dict(n=n,accepted=k,error_accepted=int(np.sum(error&accepted)),
        harm_accepted=int(np.sum(harm&accepted)),
        clean_correct_accepted=int(np.sum(clean_correct&accepted)),
        correct_rejected=int(np.sum((~error)&(~accepted))),
        risk=float(np.mean(error[accepted])) if k else None,
        harm_rate=float(np.mean(harm[accepted])) if k else None,
        harm_given_clean_correct=float(np.sum(harm&accepted)/max(1,np.sum(clean_correct&accepted))),
        actual_coverage=k/n)


def independent_check(w,lda,csp,x,a,lam,mixed):
    worst=0.
    for i,j,s in [(0,0,0),(1,min(1,len(a)-1),len(SNR)-1)]:
        waveform=x[i]+lam[j,s]*a[j]
        direct=power((w@waveform)[None])[0]
        relative=float(np.max(np.abs(direct-mixed[i,j,s]))/max(np.max(direct),1e-12))
        assert relative<1e-10
        direct_score=lda.decision_function(csp.transform(waveform[None]))[0]
        assert abs(direct_score-score(mixed[i,j,s],lda))<1e-9
        worst=max(worst,relative)
    return worst


def paired_summary(df,method,baseline,scope='all',coverage=.9):
    d=df[(df['scope']==scope)&(df['coverage']==coverage)&(df['evaluation']=='rank')]
    subjects=[];differences=[]
    for subject in sorted(d.subject.unique()):
        sub=d[d.subject==subject]
        a=sub[sub.method==method];b=sub[sub.method==baseline]
        if a.empty or b.empty:continue
        ra=a.error_accepted.sum()/a.accepted.sum()
        rb=b.error_accepted.sum()/b.accepted.sum()
        subjects.append(int(subject));differences.append(float(ra-rb))
    delta=np.array(differences)
    rng=np.random.default_rng(SEED)
    samples=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
    lo,hi=np.quantile(samples,[.025,.975])
    return dict(method=method,baseline=baseline,scope=scope,subjects=subjects,
        subject_differences=differences,mean_difference=float(delta.mean()),
        ci95=[float(lo),float(hi)],subjects_improved=int(np.sum(delta<0)),
        wilcoxon_p=float(wilcoxon(delta).pvalue) if np.any(delta) else 1.)


def run(repo,cache,out,subjects,offline_cached=False):
    out.mkdir(parents=True,exist_ok=True)
    mne.set_log_level('ERROR')
    bank=load_bank(repo,cache,offline_cached=offline_cached);rows=[];quality=[];diagnostics=[];checks=[];pairs=[]
    donor_fold=np.array([DEV.index(int(s))%3 for s in bank['subject']])
    lam=4.4/(bank['safety_rms'][:,None]*10**(SNR[None,:]/20))
    for subject in subjects:
        host=load_host(subject,cache);x,h,y,runs=[host[k] for k in ['decoder','high','y','run']]
        # Independent reproduction guard uses the original five-run fit.
        ncorrect=0
        for test_run in range(6):
            mask=runs!=test_run;csp,lda=fit_decoder(x[mask],y[mask]);w=csp.filters_[:8]
            z=np.einsum('kc,nct->nkt',w,x[~mask],optimize=True)
            ncorrect+=int(np.sum((score(power(z),lda)>0)==y[~mask]))
        ledger_acc=ncorrect/len(y)
        quality.append(dict(subject=subject,standard_oof_accuracy=ledger_acc,
            original_ledger=LEDGER[subject],ledger_delta=ledger_acc-LEDGER[subject]))
        print(f'S{subject} original-decoder guard {ledger_acc:.6f}, delta {ledger_acc-LEDGER[subject]:+.6f}',flush=True)
        if abs(ledger_acc-LEDGER[subject])>.02:
            pd.DataFrame(quality).to_csv(out/'clean_guard.csv',index=False)
            raise ValueError('Original decoder differs from ledger by more than 0.02; investigate preprocessing')
        for test_run in range(6):
            cal_run=(test_run+1)%6
            fit=(runs!=test_run)&(runs!=cal_run);cal=runs==cal_run;test=runs==test_run
            assert not np.any(fit&cal) and not np.any(fit&test) and not np.any(cal&test)
            csp,lda=fit_decoder(x[fit],y[fit]);w=csp.filters_[:8]
            px_fit=power(np.einsum('kc,nct->nkt',w,x[fit],optimize=True))
            high_reference=np.median(power(np.einsum('kc,nct->nkt',w,h[fit],optimize=True)),0)
            floor=np.maximum(np.quantile(px_fit,.01,axis=0)*.05,1e-12)
            joined=cal|test;idx=np.flatnonzero(joined)
            mixed,px,pa=terms(w,x[joined],bank['decoder'],lam)
            mh,_,ph=terms(w,h[joined],bank['high'],lam)
            checks.append(independent_check(w,lda,csp,x[joined],bank['decoder'],lam,mixed))
            iscal=runs[idx]==cal_run;istest=~iscal
            dirty_score=score(mixed,lda);clean_score=score(px,lda)
            yt=np.broadcast_to(y[idx,None,None],dirty_score.shape)
            error=(dirty_score>0)!=yt
            clean_ok=np.broadcast_to(((clean_score>0)==y[idx])[:,None,None],dirty_score.shape)
            harm=clean_ok&error
            for fold in range(3):
                train_art=donor_fold!=fold;test_art=~train_art
                assert not(set(bank['subject'][train_art])&set(bank['subject'][test_art]))
                ratio=np.exp(np.median(np.log(np.maximum(pa[train_art],1e-12))-
                                      np.log(np.maximum(ph[train_art],1e-12)),axis=0))
                estimated,_,est_pa=estimate_features(mixed,mh,lda,ratio,high_reference,floor)
                true_art=np.broadcast_to(pa[None,:,None,:]*lam[None,:,:,None]**2,mixed.shape)
                oracle=oracle_features(mixed,mh,lda,true_art,floor)
                calix=np.ix_(iscal,train_art,np.ones(len(SNR),bool))
                testix=np.ix_(istest,test_art,np.ones(len(SNR),bool))
                fcal=estimated[calix].reshape(-1,6);ftest=estimated[testix].reshape(-1,6)
                ocal=oracle[calix].reshape(-1,6);otest=oracle[testix].reshape(-1,6)
                ecal=error[calix].ravel();etest=error[testix].ravel()
                htest=harm[testix].ravel();ctest=clean_ok[testix].ravel()
                raw_cal=dirty_score[calix].ravel();raw_test=dirty_score[testix].ravel()
                calibrated={};evaluated={}
                # Safe scores are larger for safer commands.
                calibrated['margin']=fcal[:,0];evaluated['margin']=ftest[:,0]
                calibrated['high_power']=-fcal[:,1];evaluated['high_power']=-ftest[:,1]
                for name,columns in [('lr_confidence',[0]),('lr_confidence_high',[0,1]),
                                     ('estimated_power_risk',list(range(6)))]:
                    model=fit_risk(fcal[:,columns],ecal)
                    calibrated[name]=-model.predict_proba(fcal[:,columns])[:,1]
                    evaluated[name]=-model.predict_proba(ftest[:,columns])[:,1]
                model=fit_risk(ocal,ecal)
                calibrated['oracle_power_risk']=-model.predict_proba(ocal)[:,1]
                evaluated['oracle_power_risk']=-model.predict_proba(otest)[:,1]
                clean_f,_,_=estimate_features(px[istest],power(np.einsum('kc,nct->nkt',w,h[idx[istest]],optimize=True)),
                                              lda,ratio,high_reference,floor)
                clean_scores=dict(margin=clean_f[:,0],high_power=-clean_f[:,1])
                # Refit each identical calibrated model for clean control scoring.
                for name,columns in [('lr_confidence',[0]),('lr_confidence_high',[0,1]),
                                     ('estimated_power_risk',list(range(6)))]:
                    model=fit_risk(fcal[:,columns],ecal)
                    clean_scores[name]=-model.predict_proba(clean_f[:,columns])[:,1]
                # Deployment candidate diagnostic, not ground truth as a feature.
                pa_est=est_pa[testix].reshape(-1,8);pa_true=true_art[testix].reshape(-1,8)
                by_snr=np.tile(SNR,np.sum(istest)*np.sum(test_art))
                for snr in SNR:
                    mask=by_snr==snr
                    corr=spearmanr(np.log(pa_est[mask]+1e-12).ravel(),np.log(pa_true[mask]+1e-12).ravel()).statistic
                    diagnostics.append(dict(subject=subject,run=test_run,donor_fold=fold,snr=float(snr),
                        median_relative_power_error=float(np.median(np.abs(pa_est[mask]-pa_true[mask])/(pa_true[mask]+1e-12))),
                        log_power_spearman=float(corr)))
                for name in METHODS:
                    for scope,mask in [('all',np.ones(len(etest),bool))]+[(str(int(s)),by_snr==s) for s in SNR]:
                        for coverage in COVERAGES:
                            m=metrics(evaluated[name][mask],etest[mask],htest[mask],ctest[mask],coverage)
                            rows.append(dict(subject=subject,run=test_run,cal_run=cal_run,donor_fold=fold,
                                method=name,scope=scope,coverage=coverage,evaluation='rank',**m))
                    threshold=float(np.quantile(calibrated[name],.1))
                    m=metrics(evaluated[name],etest,htest,ctest,.9,threshold)
                    rows.append(dict(subject=subject,run=test_run,cal_run=cal_run,donor_fold=fold,
                        method=name,scope='all',coverage=.9,evaluation='calibrated_threshold',threshold=threshold,**m))
                    if name in clean_scores:
                        clean_error=(clean_score[istest]>0)!=y[idx[istest]]
                        m=metrics(clean_scores[name],clean_error,np.zeros(len(clean_error),bool),~clean_error,.9,threshold)
                        rows.append(dict(subject=subject,run=test_run,cal_run=cal_run,donor_fold=fold,
                            method=name,scope='clean',coverage=.9,evaluation='calibrated_threshold',threshold=threshold,**m))
                pairs.append(dict(subject=np.full(len(etest),subject),run=np.full(len(etest),test_run),
                    donor_fold=np.full(len(etest),fold),snr=by_snr,error=etest,harm=htest,
                    clean_correct=ctest,**evaluated))
            print(f'S{subject} run{test_run}: calibration run{cal_run}, 3 donor folds done',flush=True)
            pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
            pd.DataFrame(diagnostics).to_csv(out/'power_diagnostics.csv',index=False)
        pd.DataFrame(quality).to_csv(out/'clean_guard.csv',index=False)
    df=pd.DataFrame(rows)
    pair_out={key:np.concatenate([q[key] for q in pairs]) for key in pairs[0]}
    np.savez_compressed(out/'evaluation_pairs.npz',**pair_out)
    summ=[]
    for (name,scope,coverage,evaluation),d in df.groupby(['method','scope','coverage','evaluation']):
        summ.append(dict(method=name,scope=scope,coverage=float(coverage),evaluation=evaluation,
            accepted=int(d.accepted.sum()),n=int(d.n.sum()),
            error_risk=float(d.error_accepted.sum()/max(1,d.accepted.sum())),
            harm_risk=float(d.harm_accepted.sum()/max(1,d.accepted.sum())),
            actual_coverage=float(d.accepted.sum()/d.n.sum()),
            correct_rejected=int(d.correct_rejected.sum())))
    pd.DataFrame(summ).to_csv(out/'summary.csv',index=False)
    comparisons=[paired_summary(df,m,b,scope) for m in ['lr_confidence_high','estimated_power_risk','oracle_power_risk']
                 for b in ['margin','lr_confidence_high'] if m!=b for scope in ['all','-15']]
    manifest=dict(status='completed_exploratory',seed=SEED,host_subjects=subjects,
        donor_subjects=list(map(int,np.unique(bank['subject']))),n_donors=len(bank['id']),
        n_unique_evaluation_pairs=len(pair_out['error']),snr_db=SNR.tolist(),
        exact_power_check_max_error=float(max(checks)),
        original_donor_rms_max_relative_drift=float(bank['relative_original_rms_drift'].max()),
        clean_guard=quality,comparisons=comparisons,
        protocol_sha256=hashlib.sha256((Path(__file__).parent/'PROTOCOL.md').read_bytes()).hexdigest(),
        versions={p:importlib.metadata.version(p) for p in ['mne','numpy','scipy','scikit-learn','pandas']},
        python=platform.python_version(),offline_cached_subset=offline_cached,
        limitations=['Exploratory public-development-data analysis; no new sealed confirmation.',
            'At most two donors per person; not the full 247-donor bank.',
            'High-frequency mapping is approximate; no online or clinical safety claim.',
            'All six candidates reported; oracle unavailable at deployment.'])
    (out/'RESULT.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    p.add_argument('--subjects',type=int,nargs='+',default=list(range(1,10)))
    p.add_argument('--offline-cached',action='store_true',help='Use complete locally cached donor ranges only; no new donor downloads')
    args=p.parse_args();repo=Path(__file__).resolve().parents[2]
    run(repo,args.cache,args.out,args.subjects,args.offline_cached)
