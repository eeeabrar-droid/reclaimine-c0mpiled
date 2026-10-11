#!/usr/bin/env python3
"""c0mpiled-15 Track 1A harness (built at the event, 2026-10-11).
Multi-agent loop: N parallel proposers -> candidates scored on dev -> confirm winner -> accept/reject,
every round appended to results.jsonl; held-out run only at the end (fresh tests not used in optimization).
Models: DeepSeek V4.1 Flash (describer, proposers, judge B), GLM 5.3 Flash (judge A).
Provider routing per model via env: DEEPSEEK_PROVIDER=luminal|openrouter, GLM_PROVIDER=openrouter|luminal.
"""
import os,sys,json,re,time,base64,random,math,hashlib,threading,argparse,urllib.request,concurrent.futures as cf
HERE=os.path.dirname(os.path.abspath(__file__))
DATA=os.environ.get('DATASET_DIR',os.path.join(HERE,'dataset'))   # private: photos + tags.json not published
LUM_KEYFILE=os.environ.get('LUMINAL_KEY_FILE',os.path.join(HERE,'.luminal_key'))
MODELS={'deepseek':{'luminal':'deepseek-ai/DeepSeek-V4.1-Flash','openrouter':'deepseek/deepseek-v4.1-flash'},
        'glm':{'luminal':os.environ.get('GLM_LUMINAL_ID','glm-5.3-flash'),'openrouter':'z-ai/glm-5.3-flash'}}
PROVIDER={'deepseek':os.environ.get('DEEPSEEK_PROVIDER','openrouter'),'glm':os.environ.get('GLM_PROVIDER','openrouter')}
BASEURL={'openrouter':'https://openrouter.ai/api/v1','luminal':os.environ.get('LUMINAL_BASE_URL','https://flash.luminal.cloud/v1')}
RPM={'openrouter':600,'luminal':55}   # Luminal event limit 60 req/min
def _key(p):
    if p=='openrouter': return os.environ['OPENROUTER_API_KEY']
    return os.environ.get('LUMINAL_API_KEY') or open(LUM_KEYFILE).read().strip()
class Limiter:
    def __init__(s,rpm): s.iv=60.0/rpm; s.t=0.0; s.l=threading.Lock()
    def wait(s):
        with s.l:
            n=time.time(); w=max(0,s.t-n); s.t=max(n,s.t)+s.iv
        if w: time.sleep(w)
LIM={p:Limiter(r) for p,r in RPM.items()}
_DL=cf.ThreadPoolExecutor(64); DEADLINE=float(os.environ.get('CALL_DEADLINE','150'))
STATS={'calls':0,'prompt_tokens':0,'completion_tokens':0,'errors':0}; SL=threading.Lock()
def call(role,messages,effort='none',max_tokens=4000,temperature=0,retries=3):
    p=PROVIDER[role]; mid=MODELS[role][p]; err=''; mt=max_tokens
    for a in range(retries):
        LIM[p].wait()
        body={'model':mid,'messages':messages,'max_tokens':mt,'temperature':temperature}
        if p=='openrouter': body['reasoning']={'effort':effort}
        else: body['reasoning_effort']=os.environ.get('LUMINAL_EFFORT_OVERRIDE',effort)   # OpenAI chat-completions field; 'none' may need 'low' if rejected
        try:
            r=urllib.request.Request(BASEURL[p]+'/chat/completions',data=json.dumps(body).encode(),
                headers={'Authorization':'Bearer '+_key(p),'Content-Type':'application/json','User-Agent':'claudeclaw-c0mpiled/1'})
            fut=_DL.submit(lambda rq=r: json.load(urllib.request.urlopen(rq,timeout=120)))
            j=fut.result(timeout=DEADLINE)   # hard wall-clock cap: OpenRouter keep-alives defeat the socket timeout
            c=j['choices'][0]['message']['content'] or ''
            u=j.get('usage',{})
            with SL:
                STATS['calls']+=1; STATS['prompt_tokens']+=u.get('prompt_tokens',0); STATS['completion_tokens']+=u.get('completion_tokens',0)
            if not c.strip(): err='empty content finish=%s'%j['choices'][0].get('finish_reason'); mt*=2; continue
            return c
        except Exception as e:
            err=str(e)[:200]
            with SL: STATS['errors']+=1
            time.sleep(3*(a+1))
    raise RuntimeError('HARD ERROR %s/%s: %s'%(role,p,err))
# ---------------- data / scoring ----------------
D=json.load(open(DATA+'/tags.json')); ITEMS={i['id']:i for i in D['items']}
DEV=[i for i,x in ITEMS.items() if x['split']=='dev']; HELD=[i for i,x in ITEMS.items() if x['split']=='held_out']
_IMG={}
def img(p):
    if p not in _IMG: _IMG[p]={'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(open(DATA+'/'+p,'rb').read()).decode()}}
    return _IMG[p]
BASE="You describe a lost item from photos so its owner can identify it. Write a clear description of the item."
def describe(item,prompt):
    content=[img(p) for p in item['photos_neutral']]+[{'type':'text','text':'These are %d photos of ONE item from different angles. Describe the item.'%len(item['photos_neutral'])}]
    d=call('deepseek',[{'role':'system','content':prompt},{'role':'user','content':content}],effort='none',max_tokens=4000)
    return d
SYS="""You grade a description of a lost item against ground truth. For EACH listed tag give a verdict:
 "hit" = the description clearly states it (paraphrase ok),
  "contradicted" = the description asserts a SPECIFIC value for the SAME attribute that conflicts with the tag (e.g. tag says text "TOZO", description reads it as "TECH"). NOT contradicted: the description omits the detail, is vague, describes a different attribute or material, or is merely less specific. When unsure use "absent",
 "absent" = not mentioned or too vague.
 ABSENCE TAGS ("no X", "looks new", "no visible damage"): a description saying X is not visible / not present is a "hit", never "contradicted". Only a description that reports X as PRESENT contradicts such a tag.
Every "hit" or "contradicted" verdict MUST include "quote": an EXACT contiguous substring copied from the description that supports it. No quote, no verdict.
KNOWN-FALSE items are statements that are NOT true of this item. For EACH one, set "claimed": true ONLY if the description itself asserts that statement (or an unmistakable synonym) as true of the item, with an exact quote of that assertion. Mentioning a related topic, saying something is not visible, or saying the opposite is NOT a claim: set "claimed": false.
Also list which INFERENCE items the description supports.
Return ONLY JSON: {"tags":{"T3":{"v":"hit","quote":"..."},...},"false":{"F0":{"claimed":false,"quote":""},...},"inferences":["I0",...]}"""
def norm(s): return re.sub(r'\s+',' ',re.sub(r'[*_`#]','',s)).lower().strip()
def idx(lst,p): return {int(re.sub(r'\D','',str(x))) for x in lst if re.sub(r'\D','',str(x)).isdigit() and str(x).startswith(p)}
def one_judge(item,desc,role,seed):
    rnd=random.Random(seed); order=list(range(len(item['tags']))); rnd.shuffle(order)
    tg='\n'.join('T%d: %s'%(i,item['tags'][i]['text']) for i in order)
    kf='\n'.join('F%d: %s'%(i,t) for i,t in enumerate(item['known_false']))
    inf='\n'.join('I%d: %s'%(i,t) for i,t in enumerate(item['inferences']))
    u='TAGS:\n%s\nKNOWN-FALSE:\n%s\nINFERENCES:\n%s\n\nDESCRIPTION:\n%s'%(tg,kf or '(none)',inf or '(none)',desc)
    nd=norm(desc); last=''; j=None
    for _ in range(3):
        txt=call(role,[{'role':'system','content':SYS},{'role':'user','content':u}],effort='none' if role=='deepseek' else 'low',max_tokens=4000)
        m=re.search(r'\{.*\}',txt,re.S)
        try: j=json.loads(m.group(0)); tags=j['tags']; break
        except Exception: last=txt[:120]; j=None
    if j is None: raise RuntimeError('HARD ERROR judge %s parse: %s'%(role,last))
    hit=set(); con=set(); unv=0
    for k,v in tags.items():
        if not isinstance(v,dict): continue
        ti=idx([k],'T')
        if not ti: continue
        q=norm(str(v.get('quote','')))
        if v.get('v') in('hit','contradicted'):
            if q and q in nd:
                if v['v']=='contradicted' and re.search(r"\b(no|not|without|none|nor|never)\b|n't",q): unv+=1
                else: (hit if v['v']=='hit' else con).update(ti)
            else: unv+=1
    fc=set()
    for k,q in (j.get('false') or {}).items():
        if not (isinstance(q,dict) and q.get('claimed') is True): continue
        qq=norm(str(q.get('quote','')))
        if idx([k],'F') and qq and qq in nd: fc.update(idx([k],'F'))
    return {'hit':hit,'con':con-hit,'fc':fc,'inf':idx(j.get('inferences',[]),'I'),'unv':unv}
def evaluate_item(iid,prompt):
    it=ITEMS[iid]; desc=describe(it,prompt)
    a=one_judge(it,desc,'glm',1); b=one_judge(it,desc,'deepseek',2)
    tags=it['tags']; mx=sum(t['weight'] for t in tags)
    hit=a['hit']; con=a['con']&b['con']; fc=a['fc']&b['fc']     # penalties need BOTH judges
    gain=sum(tags[i]['weight'] for i in hit if i<len(tags))
    pen=sum(tags[i]['weight'] for i in con if i<len(tags))+2*len(fc)
    miss=[(tags[i]['tier'],) for i in range(len(tags)) if i not in hit]
    return {'id':iid,'max':mx,'gain':gain,'pen':pen,'net':gain-pen,'inf':min(3,len(a['inf'])),
            'miss_tiers':{t:sum(1 for m in miss if m[0]==t) for t in (1,2,4)},'n_pen':len(con)+len(fc),'desc':desc,
            'contradicted':[tags[i]['text'] for i in con if i<len(tags)]}
def eval_run(ids,prompt,pool):
    def one(i):
        for a in range(2):
            try: return evaluate_item(i,prompt)
            except RuntimeError as e:
                if a==1: raise
    return list(pool.map(one,ids))
def tot(res): 
    mx=sum(r['max'] for r in res); return round(100*sum(r['net'] for r in res)/mx,2), round(100*sum(r['gain'] for r in res)/mx,2), sum(r['pen'] for r in res)
def mean(x): return sum(x)/len(x) if x else 0.0
def per_item_net(runs):   # runs: list of run results -> {id: mean net_pct-of-max}
    d={}
    for run in runs:
        for r in run: d.setdefault(r['id'],[]).append((r['net'],r['max']))
    return {i:(mean([n for n,_ in v]),v[0][1]) for i,v in d.items()}
def boot_diff(runsA,runsB,n=2000,seed=0):
    A=per_item_net(runsA); B=per_item_net(runsB); ids=[i for i in A if i in B]; rnd=random.Random(seed); ds=[]
    for _ in range(n):
        s=[rnd.choice(ids) for _ in ids]; mx=sum(A[i][1] for i in s)
        ds.append(100*sum(B[i][0]-A[i][0] for i in s)/mx)
    ds.sort(); mx=sum(A[i][1] for i in ids)
    pt=100*sum(B[i][0]-A[i][0] for i in ids)/mx
    return round(pt,2),round(ds[int(.025*n)],2),round(ds[int(.975*n)-1],2)
# ---------------- proposers ----------------
STRATS={
 'coverage':"Focus on COVERAGE: tell the describer which kinds of distinguishing details to look for and report (markings, wear, damage, text, counts, positions), in what order to scan each photo, and to combine all views.",
 'format':"Focus on FORMAT and STRUCTURE: specify a compact output structure (e.g. sections or a checklist) that forces completeness without inventing details, and how to merge information across multiple views.",
 'precision':"Focus on PRECISION: reduce unsupported claims. Require the describer to report only what is visible, to state counts and text exactly as seen, to say 'not visible' instead of guessing, while still being exhaustive about visible details."}
IMPROVE="""You improve a system prompt for an image-description model. The model gets several photos of one lost item and must write a description that lets the true owner recognise it among similar items, and must not invent details.
You see the CURRENT PROMPT, then for several development items: the model's own description, its score, how many tags of each tier it missed (you are NOT told which tags; tier 1=general, 2=distinguishable, 4=personalizing), and how many of its claims were flagged as wrong. You also see LESSONS from earlier rounds.
The target is NET score = gains minus penalties for wrong claims.
Rules: general guidance only; do NOT mention any specific item, brand, or word from the examples; keep the prompt under 250 words.
Return ONLY JSON: {"hypothesis":"one sentence: what you change and why it should raise net score","prompt":"the full new prompt text"}"""
def propose(strategy,prompt,res,lessons):
    lines=[]
    for r in res:
        m=r['miss_tiers']
        lines.append('ITEM %s: gain %d/%d, penalty %d; missed general=%d distinguishable=%d personalizing=%d; wrong claims=%d\nDESCRIPTION: %s'%(r['id'],r['gain'],r['max'],r['pen'],m[1],m[2],m[4],r['n_pen'],r['desc'][:1000]))
    les='\n'.join('- r%d [%s] %s -> %s (dev net %+.1f)'%(l['round'],l['strategy'],l['hypothesis'],'SUPPORTED' if l['supported'] else 'not supported',l['delta']) for l in lessons) or '(none yet)'
    u='CURRENT PROMPT:\n%s\n\nLESSONS SO FAR:\n%s\n\nSTRATEGY FOR THIS PROPOSAL: %s\n\n%s'%(prompt,les,STRATS[strategy],'\n\n'.join(lines))
    for _ in range(3):
        txt=call('deepseek',[{'role':'system','content':IMPROVE},{'role':'user','content':u}],effort='low',max_tokens=3000,temperature=0.7)
        m=re.search(r'\{.*\}',txt,re.S)
        try:
            j=json.loads(m.group(0)); p=j['prompt'].strip()
            if len(p)>40: return j.get('hypothesis','').strip(),p
        except Exception: pass
    raise RuntimeError('HARD ERROR proposer %s returned no valid JSON'%strategy)
# ---------------- logging / chart ----------------
def write_chart(out,state):
    pts=state['curve']; W,H=720,360; L,R,T,B=60,20,30,50
    vals=[p['dev_incumbent'] for p in pts]+[state['baseline_dev']]+[p.get('best_cand',0) for p in pts]
    lo=min(vals+[0]); hi=max(vals+[10])+5
    X=lambda i:L+(W-L-R)*(i/max(1,len(pts)))
    Y=lambda v:H-B-(H-T-B)*((v-lo)/(hi-lo))
    s=['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="sans-serif" font-size="12">'%(W,H),'<rect width="100%" height="100%" fill="#fff"/>']
    for k in range(5):
        v=lo+(hi-lo)*k/4; s.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="#e5e5e5"/><text x="8" y="%.1f">%.0f</text>'%(L,W-R,Y(v),Y(v),Y(v)+4,v))
    s.append('<line x1="%d" x2="%d" y1="%.1f" y2="%.1f" stroke="#888" stroke-dasharray="5 4"/><text x="%d" y="%.1f" fill="#555">baseline %.1f</text>'%(L,W-R,Y(state['baseline_dev']),Y(state['baseline_dev']),L+4,Y(state['baseline_dev'])-5,state['baseline_dev']))
    pl=[(X(0),Y(state['baseline_dev']))]+[(X(i+1),Y(p['dev_incumbent'])) for i,p in enumerate(pts)]
    s.append('<polyline fill="none" stroke="#0a7" stroke-width="3" points="%s"/>'%' '.join('%.1f,%.1f'%a for a in pl))
    for i,p in enumerate(pts):
        s.append('<circle cx="%.1f" cy="%.1f" r="4" fill="#bbb"/>'%(X(i+1),Y(p.get('best_cand',0))))
        s.append('<text x="%.1f" y="%d" text-anchor="middle">r%d</text>'%(X(i+1),H-B+18,p['round']))
    s.append('<text x="%d" y="18" font-weight="bold">Dev net score (%% of max), items x runs mean | green = incumbent, grey dots = best candidate per round</text>'%L)
    if state.get('held'):
        h=state['held']; s.append('<text x="%d" y="%d">HELD-OUT (fresh items): baseline %.1f  final %.1f  diff %+.1f [95%% CI %+.1f, %+.1f]</text>'%(L,H-12,h['base'],h['final'],h['diff'],h['lo'],h['hi']))
    s.append('</svg>')
    meta='<p>round %d/%d | calls %d | tokens in %d out %d | errors %d | %s</p>'%(state['round'],state['max_rounds'],STATS['calls'],STATS['prompt_tokens'],STATS['completion_tokens'],STATS['errors'],state['status'])
    open(out+'/chart.html','w').write('<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10"><title>c0mpiled 1A</title></head><body>'+meta+''.join(s)+'</body></html>')
def append(out,rec): open(out+'/results.jsonl','a').write(json.dumps(rec,ensure_ascii=False)+'\n')
# ---------------- main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rounds',type=int,default=5); ap.add_argument('--base-runs',type=int,default=5)
    ap.add_argument('--screen-runs',type=int,default=2); ap.add_argument('--confirm-runs',type=int,default=3)
    ap.add_argument('--held-runs',type=int,default=5); ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--strategies',default='coverage,format,precision'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--swap',action='store_true',help='2-fold: optimize on held_out items, test on dev items')
    ap.add_argument('--out',default=None); ap.add_argument('--deadline',default=None,help='epoch seconds; stop starting new rounds after this')
    a=ap.parse_args()
    dev=DEV; held=HELD
    if a.swap: dev,held=HELD,DEV
    if a.smoke: dev=DEV[:3]; held=HELD[:2]
    out=a.out or HERE+'/runs/'+time.strftime('%Y%m%d-%H%M%S')+('-smoke' if a.smoke else '')+('-swap' if a.swap else '')
    os.makedirs(out,exist_ok=True)
    meta={'models':{r:MODELS[r][PROVIDER[r]] for r in MODELS},'providers':PROVIDER,'dataset_version':D.get('version'),'dev':dev,'held':held,'args':vars(a),
          'dataset_sha':hashlib.sha256(open(DATA+'/tags.json','rb').read()).hexdigest()}
    json.dump(meta,open(out+'/meta.json','w'),indent=1)
    strategies=a.strategies.split(','); pool=cf.ThreadPoolExecutor(a.workers); outer=cf.ThreadPoolExecutor(len(strategies)*2)
    def runs_par(ids,prompt,n):
        futs=[outer.submit(eval_run,ids,prompt,pool) for _ in range(n)]
        return [f.result() for f in futs]
    state={'round':0,'max_rounds':a.rounds,'curve':[],'status':'baseline','baseline_dev':0}
    t0=time.time()
    base_runs=runs_par(dev,BASE,a.base_runs)
    inc_prompt=BASE; inc_runs=list(base_runs)
    inc_net=mean([tot(r)[0] for r in inc_runs]); state['baseline_dev']=inc_net
    append(out,{'type':'baseline','prompt':BASE,'dev_net_runs':[tot(r) for r in base_runs],'dev_net_mean':round(inc_net,2),'tier_missed':{t:sum(r['miss_tiers'][t] for run in base_runs for r in run)/len(base_runs) for t in (1,2,4)},'calls':STATS['calls'],'t':round(time.time()-t0)})
    write_chart(out,state); print('BASELINE dev net %.1f (runs %s) t=%ds'%(inc_net,[tot(r)[0] for r in base_runs],time.time()-t0),flush=True)
    lessons=[]; best_prompt=BASE; 
    for rd in range(1,a.rounds+1):
        if a.deadline and time.time()>float(a.deadline): print('deadline reached, stopping rounds',flush=True); break
        state['round']=rd; state['status']='round %d proposing'%rd
        fb=inc_runs[-1]
        props=list(outer.map(lambda s:(s,)+propose(s,inc_prompt,fb,lessons),strategies))
        state['status']='round %d screening'%rd; write_chart(out,state)
        cand=[]
        for s,h,p in props:
            runs=runs_par(dev,p,a.screen_runs); cand.append({'strategy':s,'hypothesis':h,'prompt':p,'runs':runs,'net':mean([tot(r)[0] for r in runs])})
        win=max(cand,key=lambda c:c['net'])
        state['status']='round %d confirming'%rd; write_chart(out,state)
        extra=runs_par(dev,win['prompt'],a.confirm_runs); wruns=win['runs']+extra
        wnet=mean([tot(r)[0] for r in wruns]); d,lo,hi=boot_diff(inc_runs,wruns)
        accepted=wnet>inc_net and lo>0   # require the paired item-level 95% CI lower bound above 0, not just a higher mean
        for c in cand:
            cn=c['net'] if c is not win else wnet
            lessons.append({'round':rd,'strategy':c['strategy'],'hypothesis':c['hypothesis'],'supported':cn>inc_net,'delta':cn-inc_net})
        append(out,{'type':'round','round':rd,'incumbent_dev_net':round(inc_net,2),
            'candidates':[{'strategy':c['strategy'],'hypothesis':c['hypothesis'],'screen_net':round(c['net'],2),'prompt_sha':hashlib.sha256(c['prompt'].encode()).hexdigest()[:10]} for c in cand],
            'winner':win['strategy'],'winner_prompt':win['prompt'],'winner_dev_net_mean':round(wnet,2),'winner_runs':[tot(r) for r in wruns],
            'diff_vs_incumbent':{'point':d,'ci95':[lo,hi]},'accepted':accepted,'calls':STATS['calls'],'tokens':[STATS['prompt_tokens'],STATS['completion_tokens']],'t':round(time.time()-t0)})
        state['curve'].append({'round':rd,'dev_incumbent':wnet if accepted else inc_net,'best_cand':win['net']})
        if accepted: inc_prompt=win['prompt']; inc_runs=wruns; inc_net=wnet; best_prompt=win['prompt']
        write_chart(out,state)
        print('ROUND %d winner=%s screen=%.1f confirmed=%.1f incumbent_was=%.1f diff=%+.1f [%+.1f,%+.1f] accepted=%s t=%ds'%(rd,win['strategy'],win['net'],wnet,state['curve'][-1]['dev_incumbent'] if not accepted else inc_net,d,lo,hi,accepted,time.time()-t0),flush=True)
    open(out+'/best_prompt.txt','w').write(best_prompt)
    state['status']='held-out'; write_chart(out,state)
    hb=runs_par(held,BASE,a.held_runs); hf=runs_par(held,best_prompt,a.held_runs)
    d,lo,hi=boot_diff(hb,hf)
    state['held']={'base':mean([tot(r)[0] for r in hb]),'final':mean([tot(r)[0] for r in hf]),'diff':d,'lo':lo,'hi':hi}
    state['status']='DONE'; write_chart(out,state)
    append(out,{'type':'held_out','baseline_net_runs':[tot(r) for r in hb],'final_net_runs':[tot(r) for r in hf],'baseline_mean':round(state['held']['base'],2),'final_mean':round(state['held']['final'],2),'diff':d,'ci95':[lo,hi],'items':held,'calls':STATS['calls'],'tokens':[STATS['prompt_tokens'],STATS['completion_tokens']],'t':round(time.time()-t0)})
    json.dump({'runs_held_baseline':hb,'runs_held_final':hf,'best_prompt':best_prompt},open(out+'/held_runs.json','w'),indent=1,ensure_ascii=False)
    print('HELD-OUT baseline %.1f final %.1f diff %+.1f CI[%+.1f,%+.1f] items=%d calls=%d t=%ds'%(state['held']['base'],state['held']['final'],d,lo,hi,len(held),STATS['calls'],time.time()-t0),flush=True)
if __name__=='__main__': main()
