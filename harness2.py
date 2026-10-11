#!/usr/bin/env python3
"""Pipeline-level search (2026-10-11 3:10 PM): proposers change the describer PIPELINE, not just its wording.
Modes: prompt (one joint call) | per_photo (describe each photo, then merge) | marks_pass (joint + second pass for owner-only marks).
Reuses harness.py scoring/judges/stats unchanged."""
import os,sys,json,re,time,hashlib,argparse,concurrent.futures as cf
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import harness as H
MERGE_DEFAULT="You merge photo-by-photo notes about ONE item into a single description. Keep every specific detail that appears in any photo (text, marks, wear, counts, colors, positions). Do not add anything that is not in the notes. If notes conflict, keep the more specific one only if another photo supports it."
MARKS_DEFAULT="Look at the photos again. List ONLY owner-identifying details that are visible: scratches, stickers, wear, damage, handwriting, exact printed text, counts, unusual attachments. One short line each. Never guess. If none are visible, write 'none visible'."
def describe_cfg(item,cfg):
    ph=item['photos_neutral']
    if cfg['mode']=='prompt': return _orig(item,cfg['prompt'])
    if cfg['mode']=='per_photo':
        parts=[H.call('deepseek',[{'role':'system','content':cfg['prompt']},{'role':'user','content':[H.img(p),{'type':'text','text':'This is photo %d of %d of ONE item. Describe everything visible in this photo.'%(k+1,len(ph))}]}],effort='none',max_tokens=2500) for k,p in enumerate(ph)]
        return H.call('deepseek',[{'role':'system','content':cfg['aux']},{'role':'user','content':'Photo-by-photo notes of ONE item:\n\n'+'\n\n'.join('PHOTO %d:\n%s'%(i+1,x) for i,x in enumerate(parts))+'\n\nWrite the final description.'}],effort='none',max_tokens=3000)
    if cfg['mode']=='marks_pass':
        d=_orig(item,cfg['prompt'])
        extra=H.call('deepseek',[{'role':'system','content':cfg['aux']},{'role':'user','content':[H.img(p) for p in ph]+[{'type':'text','text':'Draft description of this item:\n'+d+'\n\nFollow your instructions.'}]}],effort='none',max_tokens=1500)
        return d+'\n\nOwner-identifying marks:\n'+extra
    raise ValueError(cfg['mode'])
_orig=H.describe; H.describe=describe_cfg
PSTRAT={'prompt':"Pipeline: ONE call sees all photos and writes the description. You write 'prompt' (system prompt). Set 'aux' to \"\".",
 'per_photo':"Pipeline: the describer sees ONE photo at a time (your 'prompt' is its system prompt), then a merge model combines the per-photo notes (your 'aux' is the merge system prompt). Use this to catch details visible in only one view.",
 'marks_pass':"Pipeline: ONE call writes the description with your 'prompt'; then a SECOND call sees the photos + that draft and lists owner-identifying marks with your 'aux' prompt; both are concatenated. Use this to target personalizing details without bloating the main description."}
IMPROVE2=H.IMPROVE.replace('Return ONLY JSON: {"hypothesis":"one sentence: what you change and why it should raise net score","prompt":"the full new prompt text"}',
 'Return ONLY JSON: {"hypothesis":"one sentence: what you change and why it should raise net score","prompt":"system prompt for the describer","aux":"second prompt (merge or marks pass), or empty"}')
def propose2(mode,cfg,res,lessons):
    lines=[]
    for r in res:
        m=r['miss_tiers']; lines.append('ITEM %s: gain %d/%d, penalty %d; missed general=%d distinguishable=%d personalizing=%d; wrong claims=%d\nDESCRIPTION: %s'%(r['id'],r['gain'],r['max'],r['pen'],m[1],m[2],m[4],r['n_pen'],r['desc'][:1000]))
    les='\n'.join('- r%d [%s] %s -> %s (dev net %+.1f)'%(l['round'],l['strategy'],l['hypothesis'],'SUPPORTED' if l['supported'] else 'not supported',l['delta']) for l in lessons) or '(none yet)'
    u='CURRENT PIPELINE: mode=%s\nCURRENT PROMPT:\n%s\nCURRENT AUX PROMPT:\n%s\n\nLESSONS SO FAR:\n%s\n\nYOUR PIPELINE FOR THIS PROPOSAL: %s\n\n%s'%(cfg['mode'],cfg['prompt'],cfg.get('aux') or '(none)',les,PSTRAT[mode],'\n\n'.join(lines))
    for _ in range(3):
        txt=H.call('deepseek',[{'role':'system','content':IMPROVE2},{'role':'user','content':u}],effort='low',max_tokens=3000,temperature=0.7)
        m=re.search(r'\{.*\}',txt,re.S)
        try:
            j=json.loads(m.group(0)); p=j['prompt'].strip(); aux=(j.get('aux') or '').strip()
            if mode=='per_photo' and len(aux)<40: aux=MERGE_DEFAULT
            if mode=='marks_pass' and len(aux)<40: aux=MARKS_DEFAULT
            if len(p)>40: return j.get('hypothesis','').strip(),{'mode':mode,'prompt':p,'aux':aux if mode!='prompt' else ''}
        except Exception: pass
    raise RuntimeError('HARD ERROR proposer %s returned no valid JSON'%mode)
def sha(cfg): return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:10]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rounds',type=int,default=3); ap.add_argument('--base-runs',type=int,default=5)
    ap.add_argument('--screen-runs',type=int,default=2); ap.add_argument('--confirm-runs',type=int,default=3); ap.add_argument('--held-runs',type=int,default=5)
    ap.add_argument('--workers',type=int,default=12); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    dev,held=(H.DEV[:3],H.HELD[:2]) if a.smoke else (H.DEV,H.HELD)
    out=H.HERE+'/runs/'+time.strftime('%Y%m%d-%H%M%S')+'-pipeline'+('-smoke' if a.smoke else ''); os.makedirs(out,exist_ok=True)
    json.dump({'models':{r:H.MODELS[r][H.PROVIDER[r]] for r in H.MODELS},'providers':H.PROVIDER,'dev':dev,'held':held,'args':vars(a),'search':'pipeline modes '+','.join(PSTRAT)},open(out+'/meta.json','w'),indent=1)
    pool=cf.ThreadPoolExecutor(a.workers); outer=cf.ThreadPoolExecutor(6)
    def runs_par(ids,cfg,n): return [f.result() for f in [outer.submit(H.eval_run,ids,cfg,pool) for _ in range(n)]]
    mean=H.mean; tot=H.tot; t0=time.time()
    BASECFG={'mode':'prompt','prompt':H.BASE,'aux':''}
    state={'round':0,'max_rounds':a.rounds,'curve':[],'status':'baseline','baseline_dev':0}
    inc=BASECFG; inc_runs=runs_par(dev,inc,a.base_runs); inc_net=mean([tot(r)[0] for r in inc_runs]); state['baseline_dev']=inc_net
    H.append(out,{'type':'baseline','prompt':H.BASE,'dev_net_runs':[tot(r) for r in inc_runs],'dev_net_mean':round(inc_net,2),'tier_missed':{t:sum(r['miss_tiers'][t] for run in inc_runs for r in run)/len(inc_runs) for t in (1,2,4)},'calls':H.STATS['calls'],'t':round(time.time()-t0)})
    H.write_chart(out,state); print('BASELINE dev net %.1f (runs %s) t=%ds'%(inc_net,[tot(r)[0] for r in inc_runs],time.time()-t0),flush=True)
    lessons=[]; best=inc
    for rd in range(1,a.rounds+1):
        state['round']=rd; fb=inc_runs[-1]
        props=list(outer.map(lambda m:(m,)+propose2(m,inc,fb,lessons),list(PSTRAT)))
        cand=[]
        for m,h,cfg in props:
            runs=runs_par(dev,cfg,a.screen_runs); cand.append({'strategy':m,'hypothesis':h,'cfg':cfg,'runs':runs,'net':mean([tot(r)[0] for r in runs])})
        win=max(cand,key=lambda c:c['net']); wruns=win['runs']+runs_par(dev,win['cfg'],a.confirm_runs)
        wnet=mean([tot(r)[0] for r in wruns]); d,lo,hi=H.boot_diff(inc_runs,wruns); accepted=wnet>inc_net and lo>0
        for c in cand:
            cn=c['net'] if c is not win else wnet; lessons.append({'round':rd,'strategy':c['strategy'],'hypothesis':c['hypothesis'],'supported':cn>inc_net,'delta':cn-inc_net})
        H.append(out,{'type':'round','round':rd,'incumbent_dev_net':round(inc_net,2),'incumbent_mode':inc['mode'],
            'candidates':[{'strategy':c['strategy'],'hypothesis':c['hypothesis'],'screen_net':round(c['net'],2),'prompt_sha':sha(c['cfg'])} for c in cand],
            'winner':win['strategy'],'winner_cfg':win['cfg'],'winner_dev_net_mean':round(wnet,2),'winner_runs':[tot(r) for r in wruns],
            'diff_vs_incumbent':{'point':d,'ci95':[lo,hi]},'accepted':accepted,'calls':H.STATS['calls'],'tokens':[H.STATS['prompt_tokens'],H.STATS['completion_tokens']],'t':round(time.time()-t0)})
        state['curve'].append({'round':rd,'dev_incumbent':wnet if accepted else inc_net,'best_cand':win['net']})
        print('ROUND %d winner=%s screen=%.1f confirmed=%.1f incumbent=%.1f diff=%+.1f [%+.1f,%+.1f] accepted=%s t=%ds'%(rd,win['strategy'],win['net'],wnet,inc_net,d,lo,hi,accepted,time.time()-t0),flush=True)
        if accepted: inc=win['cfg']; inc_runs=wruns; inc_net=wnet; best=inc
        H.write_chart(out,state)
    json.dump(best,open(out+'/best_cfg.json','w'),indent=1)
    hb=runs_par(held,BASECFG,a.held_runs); hf=runs_par(held,best,a.held_runs); d,lo,hi=H.boot_diff(hb,hf)
    state['held']={'base':mean([tot(r)[0] for r in hb]),'final':mean([tot(r)[0] for r in hf]),'diff':d,'lo':lo,'hi':hi}; state['status']='DONE'; H.write_chart(out,state)
    H.append(out,{'type':'held_out','baseline_net_runs':[tot(r) for r in hb],'final_net_runs':[tot(r) for r in hf],'baseline_mean':round(state['held']['base'],2),'final_mean':round(state['held']['final'],2),'diff':d,'ci95':[lo,hi],'items':held,'best_mode':best['mode'],'calls':H.STATS['calls'],'tokens':[H.STATS['prompt_tokens'],H.STATS['completion_tokens']],'t':round(time.time()-t0)})
    json.dump({'runs_held_baseline':hb,'runs_held_final':hf,'best_prompt':best},open(out+'/held_runs.json','w'),indent=1,ensure_ascii=False)
    print('HELD-OUT baseline %.1f final %.1f diff %+.1f CI[%+.1f,%+.1f] best_mode=%s items=%d calls=%d t=%ds'%(state['held']['base'],state['held']['final'],d,lo,hi,best['mode'],len(held),H.STATS['calls'],time.time()-t0),flush=True)
if __name__=='__main__': main()
