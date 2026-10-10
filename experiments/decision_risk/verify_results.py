"""Audit paired outcomes and both person axes without rerunning fitted models."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import DEV,load_bank,load_host
from run import SNR


def audit(repo,cache,result_root=None,full_bank=False):
    source_root=Path(__file__).parent
    root=Path(result_root) if result_root else source_root
    bank=load_bank(repo,cache,offline_cached=not full_bank)
    folds=np.array([DEV.index(int(s))%3 for s in bank['subject']])
    first=dict(np.load(root/'results/evaluation_pairs.npz'))
    second=dict(np.load(root/'results_augmentation/evaluation_pairs.npz'))
    third=dict(np.load(root/'results_switching/evaluation_pairs.npz'))
    for key in ['subject','run','donor_fold','snr']:
        assert np.array_equal(first[key],second[key]) and np.array_equal(first[key],third[key])
    assert np.array_equal(first['error'],second['original_decoder_error'])
    assert np.array_equal(first['error'],third['original_decoder_error'])
    for method in ['original_decoder','real_artifact_augmented','estimated_corrected_augmented']:
        assert np.array_equal(second[method+'_error'],third[method+'_error'])
        assert np.allclose(second[method+'_safe'],third[method+'_safe'],atol=1e-12)
    subjects=[];trials=[];donors=[];donor_people=[];snrs=[]
    for subject in range(1,10):
        host=load_host(subject,cache)
        for run in range(6):
            idx=np.flatnonzero(host['run']==run)
            for fold in range(3):
                donor=np.flatnonzero(folds==fold)
                n=len(idx)*len(donor)*len(SNR)
                subjects.extend([subject]*n)
                trials.extend(np.repeat(idx,len(donor)*len(SNR)))
                di=np.tile(np.repeat(donor,len(SNR)),len(idx))
                donors.extend(di);donor_people.extend(bank['subject'][di])
                snrs.extend(np.tile(SNR,len(idx)*len(donor)))
    subjects=np.array(subjects);trials=np.array(trials);donors=np.array(donors)
    donor_people=np.array(donor_people);snrs=np.array(snrs)
    assert np.array_equal(subjects,third['subject']) and np.array_equal(snrs,third['snr'])
    keys=np.c_[subjects,trials,donors,snrs]
    assert len(np.unique(keys,axis=0))==len(keys)==9*144*len(bank['id'])*len(SNR)
    # Supplemental source keys reconstruct the deterministic saved traversal;
    # this does not alter scores, fitted models, labels or outcome data.
    np.savez_compressed(root/'results_switching/PAIR_KEYS.npz',subject=subjects,
                        trial_index=trials,donor_index=donors,donor_subject=donor_people,snr=snrs)
    estimates=[];rng=np.random.default_rng(20261010)
    artifact_people=np.unique(bank['subject']);n_people=len(artifact_people)
    host_draw=rng.integers(0,9,size=(10000,9));donor_draw=rng.integers(0,n_people,size=(10000,n_people))
    for method in ['real_artifact_augmented','estimated_corrected_augmented',
                   'high95_switch_augmented','high95_switch_corrected']:
        delta=third[method+'_error'].astype(float)-third['original_decoder_error'].astype(float)
        matrix=np.array([[delta[(subjects==s)&(donor_people==d)].mean()
                          for d in artifact_people] for s in range(1,10)])
        samples=matrix[host_draw[:,:,None],donor_draw[:,None,:]].mean((1,2))
        ci=np.quantile(samples,[.025,.975])
        eligible=subjects[np.isin(subjects,[1,3,4,6,7,8,9])]
        emask=np.isin(subjects,[1,3,4,6,7,8,9])
        estimates.append(dict(method=method,mean_error_difference=float(delta.mean()),
            host_and_donor_bootstrap_ci95=ci.tolist(),host_donor_matrix=matrix.tolist(),
            eligible_cohort_error_difference=float(delta[emask].mean())))
    selector_check=None
    if (root/'results_selector/evaluation_pairs.npz').exists():
        p=dict(np.load(root/'results_selector/evaluation_pairs.npz'))
        source_keys=np.c_[p['subject'],p['trial_index'],p['donor_subject'],p['snr']]
        # Two donor windows per person have equal person keys; their stored
        # traversal order and fold checks distinguish the donor episodes.
        old=np.isin(p['snr'],SNR)
        for method in ['original_decoder','real_artifact_augmented']:
            assert np.array_equal(p[method+'_error'][old],third[method+'_error'])
        reported=pd.read_csv(root/'results_selector/summary.csv')
        for row in reported.itertuples():
            mask=p['snr']==row.snr
            assert int(mask.sum())==row.n
            assert int(p[row.method+'_error'][mask].sum())==row.errors
            assert int(p[row.method+'_harm'][mask].sum())==row.harms
        selector_check=[]
        for scope,mask in [('original_grid',old),('weak_grid',~old),('nine_snr_grid',np.ones(len(old),bool))]:
            delta=p['calibrated_decoder_choice_error'].astype(float)-p['original_decoder_error'].astype(float)
            matrix=np.array([[delta[mask&(p['subject']==s)&(p['donor_subject']==d)].mean()
                              for d in artifact_people] for s in range(1,10)])
            samples=matrix[host_draw[:,:,None],donor_draw[:,None,:]].mean((1,2))
            selector_check.append(dict(scope=scope,mean_error_difference=float(delta[mask].mean()),
                host_and_donor_bootstrap_ci95=np.quantile(samples,[.025,.975]).tolist()))
    result=dict(status='passed',n_unique_source_trial_donor_snr_pairs=len(keys),
        paired_original_outcomes_equal_across_three_stages=True,
        unchanged_stage_two_outcomes_in_stage_three=True,
        independent_people=dict(host=9,artifact=n_people),multiway_bootstrap=estimates,
        selector_summary_and_control_vectors_passed=selector_check is not None,
        calibrated_selector_diagnostics=selector_check,
        warning='Bootstrap intervals describe an exploratory fixed acquisition subset; no independent confirmation or novelty claim.',
        protocol_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_root.glob('*PROTOCOL.md')},
        acquisition_amendment_sha256=hashlib.sha256((source_root/'AMENDMENT_ACQUISITION.md').read_bytes()).hexdigest())
    (root/'RESULTS_AUDIT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--result-root',type=Path);p.add_argument('--full-bank',action='store_true');args=p.parse_args()
    audit(Path(__file__).resolve().parents[2],args.cache,args.result_root,args.full_bank)
