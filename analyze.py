#!/usr/bin/env python3
"""Overview of a harness run: rounds, accept/reject, and held-out broken down by tag tier (1 general, 2 distinguishable, 4 personalizing)."""
import json,sys,random,os
run=sys.argv[1]
recs=[json.loads(l) for l in open(run+'/results.jsonl')]
H=json.load(open(run+'/held_runs.json'))
def tier_recall(runs):   # per item: mean over runs of (tier hits / tier total); also contradictions count
    out={}
    for t in (1,2,4):
        per={}
        for run_ in runs:
            for r in run_:
                tot=None
                per.setdefault(r['id'],[]).append(r['miss_tiers'][str(t)] if str(t) in r['miss_tiers'] else r['miss_tiers'][t])
        out[t]=per
    return out
DATA=os.environ.get('DATASET_DIR',os.path.join(os.path.dirname(os.path.abspath(__file__)),'dataset'))
ITEMS={i['id']:i for i in json.load(open(DATA+'/tags.json'))['items']}
def tier_tot(i,t): return sum(1 for x in ITEMS[i]['tags'] if x['tier']==t)
def recall(runs,t):   # dict item -> mean recall of tier t
    d={}
    for run_ in runs:
        for r in run_:
            n=tier_tot(r['id'],t)
            if n==0: continue
            m=r['miss_tiers'].get(str(t),r['miss_tiers'].get(t))
            d.setdefault(r['id'],[]).append((n-m)/n)
    return {i:sum(v)/len(v) for i,v in d.items()}
def boot(A,B,n=2000):
    ids=[i for i in A if i in B]; rnd=random.Random(0); ds=[]
    for _ in range(n):
        s=[rnd.choice(ids) for _ in ids]; ds.append(100*sum(B[i]-A[i] for i in s)/len(s))
    ds.sort(); pt=100*sum(B[i]-A[i] for i in ids)/len(ids)
    return pt,ds[int(.025*n)],ds[int(.975*n)-1],len(ids)
def pen(runs): 
    return sum(r['pen'] for run_ in runs for r in run_)/len(runs), sum(r['n_pen'] for run_ in runs for r in run_)/len(runs)
L=[]
b=[r for r in recs if r['type']=='baseline'][0]
L.append('BASELINE dev net %.1f (5 runs %s)'%(b['dev_net_mean'],', '.join('%.0f'%x[0] for x in b['dev_net_runs'])))
for r in recs:
    if r['type']!='round': continue
    c='; '.join('%s %.1f'%(x['strategy'],x['screen_net']) for x in r['candidates'])
    L.append('R%d: candidates [%s] -> winner %s confirmed %.1f vs incumbent %.1f, diff %+.1f CI[%+.1f,%+.1f] %s'%(r['round'],c,r['winner'],r['winner_dev_net_mean'],r['incumbent_dev_net'],r['diff_vs_incumbent']['point'],*r['diff_vs_incumbent']['ci95'],'ACCEPTED' if r['accepted'] else 'rejected'))
h=[r for r in recs if r['type']=='held_out'][0]
L.append('HELD-OUT net: baseline %.1f -> final %.1f, diff %+.1f CI[%+.1f,%+.1f] (%d items x %d runs)'%(h['baseline_mean'],h['final_mean'],h['diff'],*h['ci95'],len(h['items']),len(h['final_net_runs'])))
hb,hf=H['runs_held_baseline'],H['runs_held_final']
names={1:'tier1 general',2:'tier2 distinguishable',4:'tier4 personalizing'}
for t in (1,2,4):
    A=recall(hb,t); B=recall(hf,t)
    if not A: continue
    pt,lo,hi,n=boot(A,B)
    L.append('HELD-OUT recall %s: %.0f%% -> %.0f%%, diff %+.1f pts CI[%+.1f,%+.1f] (%d items)'%(names[t],100*sum(A.values())/len(A),100*sum(B.values())/len(B),pt,lo,hi,n))
pb,nb=pen(hb); pf,nf=pen(hf)
L.append('HELD-OUT wrong claims per run: baseline %.1f (%.1f pts) -> final %.1f (%.1f pts)'%(nb,pb,nf,pf))
L.append('calls %d, tokens in/out %s, wall %d min'%(h['calls'],h['tokens'],h['t']//60))
print('\n'.join(L))
