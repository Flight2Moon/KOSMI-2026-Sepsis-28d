#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, signal, time
from pathlib import Path
import psutil
from los_baseline.resource_manager import nvidia_snapshot

STOP=False
def _stop(*_):
    global STOP; STOP=True

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--interval',type=float,default=2.0)
    a=p.parse_args(); a.output.parent.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,_stop); signal.signal(signal.SIGINT,_stop)
    fields=['timestamp_epoch','cpu_percent','ram_used_gb','ram_available_gb','ram_percent','gpu_index','gpu_util_percent','gpu_mem_used_gb','gpu_mem_free_gb']
    with a.output.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if f.tell()==0: w.writeheader()
        psutil.cpu_percent(interval=None)
        while not STOP:
            cpu=psutil.cpu_percent(interval=None); m=psutil.virtual_memory(); gpus=nvidia_snapshot() or [None]
            for g in gpus:
                w.writerow({'timestamp_epoch':time.time(),'cpu_percent':cpu,'ram_used_gb':m.used/1024**3,'ram_available_gb':m.available/1024**3,'ram_percent':m.percent,'gpu_index':'' if g is None else g['index'],'gpu_util_percent':'' if g is None else g['utilization_percent'],'gpu_mem_used_gb':'' if g is None else g['memory_used_gb'],'gpu_mem_free_gb':'' if g is None else g['memory_free_gb']})
            f.flush(); time.sleep(max(.5,a.interval))
    return 0
if __name__=='__main__': raise SystemExit(main())
