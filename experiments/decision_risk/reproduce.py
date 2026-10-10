"""One entry point for a cached pilot or a larger public-data extension."""
import argparse
import os
from pathlib import Path

# Set before importing numerical libraries when launched as a standalone script.
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

from augmentation import run as run_augmentation
from data import prepare, acquire_pilot
from run import run as run_risk
from stress import run as run_stress
from selector import run as run_selector
from verify_results import audit


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--full-bank',action='store_true',help='Download/prepare all 24 development artifact people')
    p.add_argument('--acquire-pilot',action='store_true',help='Download exactly the six-person pilot and nine host recordings')
    args=p.parse_args();repo=Path(__file__).resolve().parents[2]
    if args.full_bank and args.acquire_pilot:p.error('Choose only one acquisition mode')
    if args.full_bank:prepare(repo,args.cache)
    elif args.acquire_pilot:acquire_pilot(repo,args.cache)
    run_risk(repo,args.cache,args.out/'results',list(range(1,10)),offline_cached=not args.full_bank)
    run_augmentation(repo,args.cache,args.out/'results_augmentation',full_bank=args.full_bank)
    run_augmentation(repo,args.cache,args.out/'results_switching',include_switch=True,full_bank=args.full_bank)
    run_stress(repo,args.cache,args.out/'results_stress',args.out/'results_switching',args.full_bank)
    run_selector(repo,args.cache,args.out/'results_selector',args.out/'results_switching',args.full_bank)
    audit(repo,args.cache,args.out,full_bank=args.full_bank)
