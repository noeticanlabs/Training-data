from __future__ import annotations

"""UDLM certified learning runtime.

Implements the principles established from CohBit-Copilot and RPAIR without
copying either runtime: candidate updates are challenged before admission;
failures are classified and retained as evidence; uncertain candidates are held;
canonical weights move only on admitted transitions.
"""

import argparse, copy, hashlib, json, math, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from .model import UDLM, UDLMConfig, BOS

EXT={'.txt','.md','.json','.jsonl','.py','.ts','.rs'}
PROMPTS=(b'The ',b'Language ',b'Memory ')

def cj(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def shab(b): return hashlib.sha256(b).hexdigest()
def model_hash(m):
    h=hashlib.sha256()
    for n,v in sorted(m.state_dict().items()):
        h.update(n.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def files(root):
    xs=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXT]; xs.sort(key=lambda p:p.relative_to(root).as_posix().lower()); return xs

def split(data,f=.2):
    if len(data)<64:return data,data
    c=max(32,min(len(data)-32,int(len(data)*(1-f)))); return data[:c],data[c:]

def batch(data,bs,ctx,device):
    c=min(ctx,len(data)-1); mx=len(data)-c-1
    ss=[random.randrange(mx+1) if mx>0 else 0 for _ in range(bs)]
    x=torch.tensor([list(data[s:s+c]) for s in ss],dtype=torch.long,device=device)
    y=torch.tensor([list(data[s+1:s+c+1]) for s in ss],dtype=torch.long,device=device)
    return x,y

@torch.no_grad()
def bpb(m,data,ctx,device,n=4):
    if len(data)<3:return float('nan')
    old=random.getstate(); random.seed(0xC0B17+len(data)); m.eval(); tot=0.; cnt=0
    try:
        for _ in range(n):
            x,y=batch(data,2,ctx,device); z,_=m(x); loss=F.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1),reduction='sum'); tot+=float(loss); cnt+=y.numel()
        return tot/max(cnt,1)/math.log(2)
    finally: random.setstate(old)

@torch.no_grad()
def health(m,p):
    st=m.initial_state(1,m.theta.device); z=None
    for t in [BOS]+list(p): z,st=m.step(torch.tensor([t],device=m.theta.device),st)
    def s(v):
        a=v.abs(); return {'mean_abs':float(a.mean()),'rms':float(v.square().mean().sqrt()),'rail_095':float((a>.95).float().mean()),'max_abs':float(a.max())}
    return {'x':s(st.x),'r':s(st.r),'h':s(st.h),'q':s(st.q),'top_probability':float(F.softmax(z,-1).max())}

def gm(out):
    if not out:return {'length':0,'unique_byte_ratio':0.,'dominant_byte_ratio':1.,'space_ratio':0.,'nul_ratio':0.,'short_cycle':True}
    counts=[out.count(b) for b in set(out)]; tail=out[-min(64,len(out)):]; cyc=False
    for p in range(1,min(8,len(tail))+1):
        pat=tail[:p]; reb=(pat*((len(tail)+p-1)//p))[:len(tail)]
        if sum(a==b for a,b in zip(tail,reb))/len(tail)>=.9:cyc=True;break
    return {'length':len(out),'unique_byte_ratio':len(set(out))/len(out),'dominant_byte_ratio':max(counts)/len(out),'space_ratio':out.count(32)/len(out),'nul_ratio':out.count(0)/len(out),'short_cycle':cyc}

@torch.no_grad()
def panel(m,n):
    r={}
    for p in PROMPTS:
        o=m.generate(p,max_new_bytes=n,greedy=True); r[p.decode()]={'text':o.decode('utf-8',errors='replace'),'hex':o.hex(),'metrics':gm(o),'state':health(m,p)}
    return r

def deg(p):
    vals=[]
    for r in p.values():
        x=r['metrics']; vals.append(x['dominant_byte_ratio']+(.25 if x['short_cycle'] else 0)+.25*max(x['space_ratio'],x['nul_ratio']))
    return sum(vals)/len(vals)

def dyn_bad(p,limit):
    rs=[]
    for prompt,row in p.items():
        for f in ('x','r'):
            q=row['state'][f]['rail_095']
            if q>limit:rs.append(f'{f}_rail>{limit:.2f}@{prompt}:{q:.3f}')
    return bool(rs),rs

def last_hash(path):
    if not path.exists():return 'GENESIS'
    ls=[x for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if not ls:return 'GENESIS'
    try:return json.loads(ls[-1]).get('receipt_hash','GENESIS')
    except:return 'GENESIS'
def receipt(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); prev=last_hash(path); body=dict(obj);body['previous_receipt_hash']=prev; rh=hashlib.sha256((prev+'|'+cj(body)).encode()).hexdigest();body['receipt_hash']=rh
    with path.open('a',encoding='utf-8') as f:f.write(cj(body)+'\n')
    return rh
def memory(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(cj(obj)+'\n')
def save(m,path,meta):
    path.parent.mkdir(parents=True,exist_ok=True);torch.save({'config':asdict(m.cfg),'state_dict':m.state_dict(),'metadata':meta},path)
def append_replay(r,l,n):
    z=r+b'\n'+l;return z[-n:] if len(z)>n else z

def train(m,opt,acq,replay,a,device):
    m.train();last=float('nan')
    for _ in range(a.steps_per_file):
        src=replay if replay and random.random()<a.replay_ratio else acq;x,y=batch(src,a.batch_size,a.context,device);z,_=m(x);loss=F.cross_entropy(z.reshape(-1,z.size(-1)),y.reshape(-1));opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),a.grad_clip);opt.step();last=float(loss/math.log(2))
    return last

def failure_class(ag,vg,rr,db,gr,a):
    if not all(math.isfinite(v) for v in (ag,vg,rr,gr)):return 'DIAGNOSTIC_COVERAGE_GAP'
    if db:return 'DYNAMIC_FAILURE'
    if rr>a.hard_replay_regression:return 'RETENTION_FAILURE'
    if gr>.20:return 'GENERATION_FAILURE'
    if ag<=0:return 'ACQUISITION_FAILURE'
    if vg<=0:return 'GENERALIZATION_FAILURE'
    return 'DIAGNOSTIC_COVERAGE_GAP'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('curriculum',type=Path);ap.add_argument('--checkpoint',type=Path,default=Path('artifacts/udlm_certified.pt'));ap.add_argument('--receipts',type=Path,default=Path('artifacts/udlm_certified_receipts.jsonl'));ap.add_argument('--capabilities',type=Path,default=Path('artifacts/udlm_capability_memory.jsonl'));ap.add_argument('--failures',type=Path,default=Path('artifacts/udlm_failure_memory.jsonl'));ap.add_argument('--held-dir',type=Path,default=Path('artifacts/held_candidates'));ap.add_argument('--report',type=Path,default=Path('artifacts/udlm_certified_report.json'));ap.add_argument('--steps-per-file',type=int,default=200);ap.add_argument('--batch-size',type=int,default=4);ap.add_argument('--context',type=int,default=128);ap.add_argument('--lr',type=float,default=.0015);ap.add_argument('--weight-decay',type=float,default=1e-4);ap.add_argument('--grad-clip',type=float,default=1.);ap.add_argument('--replay-ratio',type=float,default=.35);ap.add_argument('--replay-bytes',type=int,default=262144);ap.add_argument('--validation-fraction',type=float,default=.2);ap.add_argument('--min-acquisition-gain',type=float,default=.02);ap.add_argument('--min-validation-gain',type=float,default=.01);ap.add_argument('--max-replay-regression',type=float,default=.10);ap.add_argument('--hard-replay-regression',type=float,default=.25);ap.add_argument('--max-generation-regression',type=float,default=.12);ap.add_argument('--rail-limit',type=float,default=.90);ap.add_argument('--seed',type=int,default=1);ap.add_argument('--max-new-bytes',type=int,default=128);ap.add_argument('--device',default='cpu');a=ap.parse_args()
    random.seed(a.seed);torch.manual_seed(a.seed);device=torch.device(a.device);m=UDLM(UDLMConfig()).to(device);opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.weight_decay);replay=b'';counts={'ADMIT':0,'HOLD':0,'REJECT':0};fs=files(a.curriculum)
    print(f'UDLM certified learner | params={m.parameter_count():,} lessons={len(fs)}');print('observe -> candidate -> challenge -> ADMIT/HOLD/REJECT -> diagnose -> receipt')
    for i,p in enumerate(fs,1):
        lesson=p.read_bytes();rel=p.relative_to(a.curriculum).as_posix();acq,val=split(lesson,a.validation_fraction);ph=model_hash(m);ab=bpb(m,acq,a.context,device);vb=bpb(m,val,a.context,device);rb=bpb(m,replay,a.context,device) if len(replay)>=3 else None;pb=panel(m,a.max_new_bytes);d0=deg(pb);ms=copy.deepcopy(m.state_dict());os=copy.deepcopy(opt.state_dict());tl=train(m,opt,acq,replay,a,device);ch=model_hash(m);aa=bpb(m,acq,a.context,device);va=bpb(m,val,a.context,device);ra=bpb(m,replay,a.context,device) if len(replay)>=3 else None;pa=panel(m,a.max_new_bytes);d1=deg(pa);ag=ab-aa;vg=vb-va;rr=0. if rb is None or ra is None else ra-rb;gr=d1-d0;db,dr=dyn_bad(pa,a.rail_limit)
        catastrophic=rr>a.hard_replay_regression or gr>.20 or db or not all(math.isfinite(v) for v in (aa,va,tl));admissible=ag>=a.min_acquisition_gain and vg>=a.min_validation_gain and rr<=a.max_replay_regression and gr<=a.max_generation_regression and not db
        fc=None
        if admissible: dec='ADMIT';counts[dec]+=1;replay=append_replay(replay,lesson,a.replay_bytes);memory(a.capabilities,{'kind':'udlm-capability-memory-v1','timestamp':time.time(),'lesson':rel,'candidate_hash':ch,'acquisition_gain_bpb':ag,'independent_validation_gain_bpb':vg,'retention_regression_bpb':rr,'generation_regression':gr,'evidence':'cross_sample_supported'})
        elif not catastrophic and ag>0: dec='HOLD';counts[dec]+=1;fc=failure_class(ag,vg,rr,db,gr,a);save(m,a.held_dir/f'candidate_{i:04d}_{ch[:12]}.pt',{'status':'HELD','lesson':rel,'parent_hash':ph});m.load_state_dict(ms);opt.load_state_dict(os)
        else: dec='REJECT';counts[dec]+=1;fc=failure_class(ag,vg,rr,db,gr,a);m.load_state_dict(ms);opt.load_state_dict(os)
        if dec!='ADMIT':memory(a.failures,{'kind':'udlm-failure-memory-v1','timestamp':time.time(),'lesson':rel,'candidate_hash':ch,'failure_class':fc,'dynamic_reasons':dr,'acquisition_gain_bpb':ag,'independent_validation_gain_bpb':vg,'retention_regression_bpb':rr,'generation_regression':gr,'continuation_requirement':'localize with existing diagnostics before proposing structural repair' if fc=='DIAGNOSTIC_COVERAGE_GAP' else f'restore {fc.lower()} condition and reverify against parent capability'})
        receipt(a.receipts,{'kind':'udlm-certified-learning-receipt-v1','timestamp':time.time(),'lesson_index':i,'lesson_path':rel,'lesson_sha256':shab(lesson),'parent_checkpoint_hash':ph,'candidate_checkpoint_hash':ch,'train_bpb_last':tl,'acquisition_bpb_before':ab,'acquisition_bpb_after':aa,'validation_bpb_before':vb,'validation_bpb_after':va,'replay_bpb_before':rb,'replay_bpb_after':ra,'generation_degeneration_before':d0,'generation_degeneration_after':d1,'dynamic_failure_reasons':dr,'decision':dec,'failure_class':fc,'counts':counts.copy()});print(f'[{i:03d}/{len(fs):03d}] {dec:6s} {rel} | acq {ab:.3f}->{aa:.3f} ({ag:+.3f}) val {vb:.3f}->{va:.3f} ({vg:+.3f}) retain {rr:+.3f} gen {gr:+.3f}')
    fh=model_hash(m);save(m,a.checkpoint,{'trainer':'udlm-certified-learning-v1','counts':counts,'canonical_model_hash':fh,'last_receipt':last_hash(a.receipts)});fp=panel(m,a.max_new_bytes);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps({'canonical_model_hash':fh,'counts':counts,'generation_panel':fp,'last_receipt':last_hash(a.receipts)},indent=2,ensure_ascii=False),encoding='utf-8');print(f'saved {a.checkpoint} counts={counts}');[print(repr(k)+' -> '+repr(v['text'])) for k,v in fp.items()]
if __name__=='__main__':main()
