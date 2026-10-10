"""Public raw-data acquisition, with bounded and checked EDF range reads."""
from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat
from scipy.signal import resample_poly

COMMON19 = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4',
            'C6','CP3','CP1','CP2','CP4','Pz','POz']
BCI_INDICES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,17,19,21]
DEV = [1,2,3,4,5,6,7,8,9,10,11,13,15,16,18,20,21,22,24,26,27,28,29,30]
EXCLUDED = {12,14,17,19,23,25}
S3 = 'https://s3.amazonaws.com/openneuro.org/'
BNCI = 'https://bnci-horizon-2020.eu/database/data-sets/001-2014/'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def get_range(url, start, stop, cache):
    """Inclusive range. Abort if proxy/server silently returns the wrong bytes."""
    cache=Path(cache);cache.parent.mkdir(parents=True,exist_ok=True)
    n=stop-start+1
    if cache.exists():
        data=cache.read_bytes()
        if len(data)==n:return data
        raise ValueError(f'Invalid cached byte count: {cache}')
    for attempt in range(3):
        try:
            r=requests.get(url,headers={'Range':f'bytes={start}-{stop}'},timeout=(20,60))
            r.raise_for_status()
            if r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{stop}/'):
                raise ValueError(f'Invalid range response {r.status_code}: {r.headers.get("Content-Range")}')
            if len(r.content)!=n:raise ValueError('Truncated EDF range')
            cache.write_bytes(r.content)
            cache.with_suffix(cache.suffix+'.json').write_text(json.dumps({
                'url':url,'range':[start,stop],'content_range':r.headers['Content-Range'],
                'etag':r.headers.get('ETag'),'sha256':hashlib.sha256(r.content).hexdigest()},indent=2))
            return r.content
        except (requests.RequestException,ValueError):
            if attempt==2:raise
            time.sleep(attempt+1)


def edf_header(url, cache):
    fixed=get_range(url,0,255,Path(cache)/'fixed.bin')
    ns=int(fixed[252:256]);n_header=int(fixed[184:192])
    h=get_range(url,0,n_header-1,Path(cache)/'header.bin')
    fields={};pos=256
    for key,n in [('label',16),('transducer',80),('unit',8),('pmin',8),('pmax',8),
                  ('dmin',8),('dmax',8),('prefilter',80),('samples',8),('reserved',32)]:
        fields[key]=[h[pos+i*n:pos+(i+1)*n].decode('ascii').strip() for i in range(ns)]
        pos+=n*ns
    assert pos==n_header
    fields.update(header_bytes=n_header,record_s=float(h[244:252]),records=int(h[236:244]),
                  samples=np.array(fields['samples'],int),ns=ns)
    assert float(fields['record_s'])>0
    return fields


def read_edf_segment(url, header, start_s, stop_s, cache):
    dur=header['record_s'];first=int(math.floor(start_s/dur));last=int(math.ceil(stop_s/dur))
    assert first>=0 and last<=header['records']
    total=int(header['samples'].sum()); record_bytes=2*total
    begin=header['header_bytes']+first*record_bytes
    end=header['header_bytes']+last*record_bytes-1
    b=get_range(url,begin,end,cache)
    raw=np.frombuffer(b,dtype='<i2').reshape(last-first,total)
    offsets=np.r_[0,np.cumsum(header['samples'])]
    vals=[]
    for name in COMMON19:
        if name not in header['label']:raise ValueError(f'Missing channel {name}')
        k=header['label'].index(name)
        fs=header['samples'][k]/dur
        if fs!=1000:raise ValueError(f'Unexpected EDF frequency {fs}')
        x=raw[:,offsets[k]:offsets[k+1]].reshape(-1).astype(float)
        pmin,pmax=float(header['pmin'][k]),float(header['pmax'][k])
        dmin,dmax=float(header['dmin'][k]),float(header['dmax'][k])
        if not dmax>dmin:raise ValueError('Invalid physical calibration')
        x=(x-dmin)*(pmax-pmin)/(dmax-dmin)+pmin
        unit=header['unit'][k].lower().replace('µ','u')
        if unit=='v':x*=1e6
        elif unit=='mv':x*=1e3
        elif unit not in ('uv','μv'):raise ValueError(f'Unknown EDF unit {unit}')
        vals.append(x)
    x=np.array(vals)
    if not np.isfinite(x).all() or np.any(np.std(x,axis=1)<1e-3):
        raise ValueError('Nonfinite or flat channel')
    return x,first*dur


def make_donor(row, cache_root):
    s=int(row.artifact_subject);run=int(row.artifact_run)
    assert s in DEV and s not in EXCLUDED
    cache=Path(cache_root)/'edf'/f'P{s:02d}_R{run:02d}'
    url=S3+f'ds006386/sub-{s}/eeg/sub-{s}_task-artifact_run-{run:02d}_eeg.edf'
    header=edf_header(url,cache)
    start=float(row.annotation_start_s);stop=float(row.annotation_stop_s)
    x,origin=read_edf_segment(url,header,start,stop,cache/f'{row.donor_id}.bin')
    # This extracts the complete annotated context; filtering then cropping
    # follows the original bank. The resampling boundary is outside the core.
    a=int(round((start-origin)*1000));b=int(round((stop-origin)*1000))+1
    segment=x[:,a:b]
    emg=mne.filter.notch_filter(segment,Fs=1000,freqs=[50],verbose='ERROR')
    emg=mne.filter.filter_data(emg,1000,20,80,verbose='ERROR')
    emg-=emg.mean(0,keepdims=True)
    saf=resample_poly(emg,1,4,axis=-1)
    dec=mne.filter.filter_data(saf,250,8,32,method='iir',verbose='ERROR')
    high=mne.filter.filter_data(saf,250,40,80,method='iir',verbose='ERROR')
    core_start=(start+stop)/2-2
    k=int(round((core_start-start)*250))
    saf,dec,high=[q[:,k:k+1001] for q in (saf,dec,high)]
    assert saf.shape==dec.shape==high.shape==(19,1001)
    measured=float(np.sqrt(np.mean(saf**2)))
    drift=abs(measured-float(row.safety_rms_uv))/float(row.safety_rms_uv)
    if drift>.05:raise ValueError(f'{row.donor_id}: donor RMS drift {drift:.1%} exceeds 5%')
    return dict(id=str(row.donor_id),subject=s,decoder=dec,high=high,safety_rms=measured,
                relative_original_rms_drift=drift)


def load_bank(repo, cache_root, offline_cached=False):
    target=Path(cache_root)/('pilot_bank_cached.npz' if offline_cached else 'pilot_bank.npz')
    if target.exists():return dict(np.load(target,allow_pickle=False))
    meta=pd.read_csv(Path(repo)/'results/bank_metadata_dev.csv')
    meta=meta.sort_values(['artifact_subject','artifact_run','episode_in_run'])
    rows=meta.groupby('artifact_subject',sort=True).head(2)
    assert sorted(rows.artifact_subject.unique())==DEV
    if offline_cached:
        # Select on acquisition completeness only, before candidate outcomes.
        ready=[]
        for row in rows.itertuples():
            p=Path(cache_root)/'edf'/f'P{int(row.artifact_subject):02d}_R{int(row.artifact_run):02d}'
            ready.append(all((p/n).exists() for n in ['fixed.bin','header.bin',str(row.donor_id)+'.bin']))
        rows=rows[np.array(ready)]
        if rows.empty:raise ValueError('No complete cached EDF donors')
    result=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(make_donor,row,cache_root):row.donor_id for row in rows.itertuples()}
        for f in as_completed(futures):
            result.append(f.result());print('donor ready',futures[f],flush=True)
    result.sort(key=lambda d:d['id'])
    out={k:np.array([r[k] for r in result]) for k in result[0]}
    np.savez_compressed(target,**out)
    target.with_suffix('.json').write_text(json.dumps({'offline_cached':offline_cached,
        'selected_ids':out['id'].tolist(),'selected_subjects':np.unique(out['subject']).tolist()},indent=2))
    return out


def download_bnci(subject,cache_root):
    path=Path(cache_root)/'bnci'/f'A{subject:02d}T.mat';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>1000000:return path
    url=BNCI+path.name
    for attempt in range(3):
        try:
            r=requests.get(url,stream=True,timeout=(20,90));r.raise_for_status()
            part=path.with_suffix('.part')
            with part.open('wb') as f:
                for chunk in r.iter_content(1<<20):f.write(chunk)
            expected=int(r.headers.get('Content-Length',part.stat().st_size))
            if part.stat().st_size!=expected or expected<1000000:raise ValueError('Truncated MATLAB download')
            part.replace(path);return path
        except (requests.RequestException,ValueError):
            if attempt==2:raise
            time.sleep(attempt+1)


def load_host(subject,cache_root):
    target=Path(cache_root)/f'host_{subject}.npz'
    if target.exists():return dict(np.load(target,allow_pickle=False))
    p=download_bnci(subject,cache_root)
    runs=loadmat(p,squeeze_me=True,struct_as_record=False)['data']
    xd=[];xh=[];labels=[];groups=[]
    run_i=0
    for run in np.atleast_1d(runs):
        trials=np.atleast_1d(run.trial)
        if len(trials)==0:continue
        y=np.atleast_1d(run.y).astype(int)
        assert float(run.fs)==250 and len(trials)==len(y)
        if not np.isin(y,[1,2,3,4]).all():raise ValueError('Unexpected BCI labels')
        raw=np.asarray(run.X[:,BCI_INDICES],float).T
        raw-=raw.mean(0,keepdims=True)
        d=mne.filter.filter_data(raw,250,8,32,method='iir',verbose='ERROR')
        high=mne.filter.notch_filter(raw,Fs=250,freqs=[50],verbose='ERROR')
        high=mne.filter.filter_data(high,250,40,80,method='iir',verbose='ERROR')
        for onset,cls in zip(trials,y):
            if cls not in (1,2):continue
            # BNCI trial markers precede the motor-imagery cue by 2 seconds.
            # MOABB adds dataset.interval[0] == 2 to paradigm tmin == 0.
            onset=int(onset)-1+500
            xd.append(d[:,onset:onset+1001]);xh.append(high[:,onset:onset+1001])
            labels.append(cls-1);groups.append(run_i)
        run_i+=1
    out=dict(decoder=np.array(xd),high=np.array(xh),y=np.array(labels),run=np.array(groups))
    assert out['decoder'].shape==(144,19,1001) and len(np.unique(groups))==6
    assert np.isfinite(out['decoder']).all() and np.isfinite(out['high']).all()
    assert .3<float(np.median(np.sqrt(np.mean(out['decoder']**2,axis=(1,2)))))<300
    np.savez_compressed(target,**out)
    target.with_suffix('.json').write_text(json.dumps({'source':BNCI+p.name,'sha256':digest(p),
        'units':'uV','session':'T only','filter':'MNE IIR 8-32; notch50 and IIR40-80',
        'channels':COMMON19,'trials':144},indent=2))
    print('host ready',subject,flush=True)
    return out


def prepare(repo,cache_root):
    root=Path(cache_root);root.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(4) as pool:
        futures={pool.submit(load_host,s,root):s for s in range(1,10)}
        for f in as_completed(futures):f.result()
    return load_bank(repo,root)


def acquire_pilot(repo,cache_root):
    """Explicit opt-in acquisition of the exact six-person pilot subset."""
    root=Path(cache_root);root.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(Path(repo)/'results/bank_metadata_dev.csv')
    meta=meta.sort_values(['artifact_subject','artifact_run','episode_in_run'])
    rows=meta.groupby('artifact_subject',sort=True).head(2)
    rows=rows[rows.artifact_subject.isin([1,2,3,4,5,6])]
    assert len(rows)==12
    for row in rows.itertuples():make_donor(row,root)
    for subject in range(1,10):load_host(subject,root)
    bank=load_bank(repo,root,offline_cached=True)
    if len(bank['id'])!=12 or set(bank['subject'])!={1,2,3,4,5,6}:
        raise ValueError('Use a fresh cache for exact pilot acquisition; other donors are present')
    return bank


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);args=p.parse_args()
    mne.set_log_level('ERROR');prepare(args.repo,args.cache)
