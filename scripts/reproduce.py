"""Run the current benchmark workflow in separate processes and a fresh results tree."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PHASES={
 'learning':[['scripts/run_study.py',p] for p in ['pilot','subset_reference','screen','rr_extension','balanced_extension','spatial_extension','confirm','robustness','controls','central','data_volume']],
 'analysis':[['-m','fedecg.analyze']],
 'validation':[['-m','fedecg.validation',p] for p in ['grouped','equal_steps','subset_stability','influence']],
 'integration':[['-m','fedecg.integration']],
 'reporting':[['scripts/build_summaries.py'],['-m','fedecg.figures'],['scripts/plot_transport.py']],
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['all',*PHASES],default='all')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto');a=p.parse_args()
    selected=list(PHASES) if a.phase=='all' else [a.phase]
    for phase in selected:
        for command in PHASES[phase]:
            args=command+(['--device',a.device] if command[0]=='scripts/run_study.py' else [])
            print('Running',phase,' '.join(args),flush=True)
            subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
    print('Completed',', '.join(selected))


if __name__=='__main__':main()
