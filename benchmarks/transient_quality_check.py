"""Reproduce isolated-transient time-stretch quality and fallback checks.

Requires NumPy, SoundFile, two pbshift CLI builds, and a configured pbACC CLI.
Creates a fresh output directory containing every source, target, output and
raw detector result. Synthetic event targets test waveform/timing preservation;
golden controls test compatibility and do not certify natural audio quality.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, json, statistics as st, subprocess
import numpy as np
import soundfile as sf

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--exe',type=Path,required=True)
ap.add_argument('--baseline',type=Path,required=True)
ap.add_argument('--pbacc',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--workers',type=int,default=3)
ap.add_argument('--seed',type=int,default=91760481)
args=ap.parse_args()
candidate=args.exe.resolve();baseline=args.baseline.resolve();judge=args.pbacc.resolve()
out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
QUAL_TOL,NOI_TOL,FLMOS_TOL,PSY_TOL,ONSET_TOL=.05,.05,.03,.02,3
p={'seed': 91760481, 'rates': [8000, 12000, 16000, 22050, 44100, 48000], 'ratios': [0.25, 0.5, 0.75, 0.9, 1, 1.1, 1.25, 2, 4], 'families': ['biphasic_click', 'inharmonic_metal', 'noise_decay', 'harmonic_decay'], 'controls': ['background', 'long_event', 'slow_rise', 'sustain', 'collision', 'dense'], 'seconds': 2.9, 'source_count': 60, 'cases': 540, 'ground_truth': 'Positive compact events: move known joint absolute-sample peak anchors by ratio, retain all exact float32 event samples and stereo timing. Collisions have known packet-sum targets; they are fallback controls at ratios producing intersecting supports. Background/long/slow/sustained/dense controls use baseline golden output for compatibility only, not a natural-quality reference. All cases retained.', 'gates': 'No new detector failure or per-case QUAL-.05 / NOI-.05 / FL-.03 / PSY+.02 / verified badonset+3. Packet NCC drop<=.01; absolute packet energy error to known target notworse>1dB; ghost energy notworse>1dB above-80dB; finite/rate/channels/length/peak. Mean known-target QUAL mustgain>=.002 OR mean ghost reduction>=3dB with>=6 baseline>=-50dB cases gaining>=6dB. Admitted outputs exactlymatchknown targets; all fallback controls exactlymatchbaseline. Allcasesretained.'}
p.update(seed=args.seed, baseline_sha256=sha(baseline), candidate_sha256=sha(candidate), judge_sha256=sha(judge))
p['harness_sha256']=sha(Path(__file__))
p['runtime_files']={str(q.relative_to(judge.parent)):sha(q) for pattern in ['*.dll','calib/*.json','calib/*.txt','models/*.onnx'] for q in judge.parent.glob(pattern)}
(out/'protocol.json').write_text(json.dumps(p,indent=2))

def g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d

def extract(js):
    det = js.get('detectors', {})

    def val(k, *p):
        if g(det, k, 'status') != 'ok':
            return None
        return g(det, k, 'value', *p)
    return {'verdict': js.get('verdict'), 'reasons': js.get('reasons', []), 'qual': val('FR-QUAL', 'mos_lqo'), 'flmos': val('FLMOS', 'mos'), 'flmos_ref': val('FLMOS', 'vs_ref', 'ref'), 'noi': val('NISQA-NOI', 'noi'), 'psy_rdf': val('PSY-MASK', 'rel_disturbed_frames'), 'psy_nmr': val('PSY-MASK', 'total_nmr_db'), 'onset_recall': val('FR-ONSET', 'recall'), 'onset_prec': val('FR-ONSET', 'precision'), 'birdie': val('FR-SPEC', 'birdie_index'), 'gain_db': val('FR-LEVEL', 'gain_db_normal'), 'f0_err': val('B-F0', 'median_err_cents'), 'len_err': val('B-LEN', 'err_pct'), 'formant_err': val('FR-FORMANT', 'scale_err_cents'), 'lsd_res': val('FR-FORMANT', 'lsd_db_residual'), 'drops': val('FR-DROP', 'events'), 'onset_counts': g(det, 'FR-ONSET', 'value', 'counts') if g(det, 'FR-ONSET', 'verified') else None, 'onset_counts_hint': g(det, 'FR-ONSET', 'value', 'counts'), 'peak_dbfs': g(det, 'NR-LEVEL', 'value', 'peak_dbfs'), 'status': {k: v.get('status') for k, v in det.items()}, 'verified': {k: v.get('verified', False) for k, v in det.items()}}

def det_of(reason):
    return reason.split(':', 1)[0]

def onset_bad(r):
    c = r.get('onset_counts') or {}
    return c.get('extra', 0) + c.get('missed', 0) + c.get('pre_echo', 0) + c.get('weakened', 0)

def issues(a, b):
    out = []
    ra = set(map(det_of, a['reasons']))
    rb = set(map(det_of, b['reasons']))
    for d in sorted(rb - ra):
        out.append(('new-fail:' + d, ''))
    for m, tol, sign in [('qual', QUAL_TOL, 1), ('noi', NOI_TOL, 1), ('flmos', FLMOS_TOL, 1), ('psy_rdf', PSY_TOL, -1)]:
        x, y = (a.get(m), b.get(m))
        if x is not None and y is not None and (sign * (y - x) < -tol):
            out.append((m, f'{x:.3f}->{y:.3f}'))
    oa, ob = (onset_bad(a), onset_bad(b))
    if ob - oa >= ONSET_TOL:
        out.append(('onset', f'{oa}->{ob}'))
    return out

def totals(run, keys):

    def mean(m):
        xs = [run[k].get(m) for k in keys if run[k].get(m) is not None]
        return st.mean(xs) if xs else float('nan')
    return {'fail_cases': sum((run[k].get('verdict') == 'fail' for k in keys)), 'fail_reasons': sum((len(run[k].get('reasons', [])) for k in keys if run[k].get('verdict') == 'fail')), 'qual': mean('qual'), 'noi': mean('noi'), 'flmos': mean('flmos')}

rng=np.random.default_rng(p['seed']);rows=[]
for sr in p['rates']:
 for family in p['families']+p['controls']:
  idx=len(rows);ch=1+idx%2;duration=p['seconds'];n=round(duration*sr);width={'biphasic_click':.0045,'inharmonic_metal':.023,'noise_decay':.047,'harmonic_decay':.071,'background':.035,'long_event':.160,'slow_rise':.070,'sustain':.060,'collision':.035,'dense':.035}[family]
  if family in p['families']:width*=rng.uniform(.9,1.05)
  count=round(width*sr);t=np.arange(count)/sr;envelope=np.exp(-t/(width/5))*np.minimum(1,t/.00025);packet=[]
  for c in range(ch):
   if family=='biphasic_click':v=np.exp(-((t-.0009)/.00032)**2)-.37*np.exp(-((t-.0018)/.00045)**2)
   elif family in ['noise_decay','background','collision','dense']:v=rng.normal(size=count)*envelope
   elif family=='inharmonic_metal':v=sum(np.sin(2*np.pi*f*t+rng.uniform(-np.pi,np.pi))*envelope for f in np.minimum(sr*.43,np.array([731,1843,3271,5933])*rng.uniform(.8,1.12)))
   elif family in ['harmonic_decay','long_event']:
    f0=rng.uniform(91,291);v=sum(np.sin(2*np.pi*f0*k*t+.18*k+.23*c)/k for k in range(1,min(14,int(sr*.44/f0))))*envelope
   elif family=='slow_rise':v=np.sin(2*np.pi*min(sr*.3,1319)*t+.23*c)*np.sin(np.pi*np.arange(count)/(count-1))**4
   else:
    v=np.sin(2*np.pi*min(sr*.3,2117)*t+.11*c);ramp=max(1,round(.001*sr));v[:ramp]*=np.linspace(0,1,ramp);v[-ramp:]*=np.linspace(1,0,ramp)
   if family not in ['slow_rise','sustain']:v[-max(2,round(.001*sr)):]*=np.linspace(1,0,max(2,round(.001*sr)))
   packet.append(v*(1 if c==0 else .71))
  packet=np.array(packet).T;packet*=.38/np.max(abs(packet));events=[round((.31+j*.43+rng.uniform(-.015,.015))*sr) for j in range(6)]
  if family=='collision':events=[round((.30+j*.060)*sr) for j in range(35)]
  if family=='dense':events=[round((.25+j*.040)*sr) for j in range(60)]
  x=np.zeros((n,ch));spans=[]
  for j,at in enumerate(events):
   end=at+len(packet);assert end<n;gain=rng.uniform(.65,1.0);x[at:end]+=packet*gain;spans.append((at,end))
  if family=='background':
   clock=np.arange(n)/sr
   for c in range(ch):x[:,c]+=.003*np.sin(2*np.pi*173*clock+.2*c)+.002*np.sin(2*np.pi*317*clock+.1*c)
  source=out/f'source_{idx:02d}.wav';sf.write(source,x,sr,subtype='FLOAT');x,_=sf.read(source,always_2d=True);targets={}
  for ratio in p['ratios']:
   m=int(np.floor(n*ratio+.5));y=np.zeros((m,ch));mapped=[];fallback=family in ['background','long_event','slow_rise','sustain','dense'];last=0
   if not fallback:
    for begin,end in spans:
     a=begin+int(np.argmax(np.max(abs(x[begin:end]),axis=1)));start=int(np.floor(a*ratio+.5))-(a-begin);stop=start+(end-begin)
     if start<last or start<0 or stop>m:fallback=True
     if start>=0 and stop<=m:y[start:stop]+=x[begin:end]
     mapped.append([start,stop]);last=stop
   reference=out/f'target_{idx:02d}_s{ratio:g}.wav'
   if not fallback or family=='collision':sf.write(reference,y,sr,subtype='FLOAT')
   targets[str(ratio)]={'path':str(reference),'fallback_expected':fallback,'reference_kind':'known_events' if family in p['families']+['collision'] else 'golden_compatibility','spans':mapped}
  rows.append({'id':idx,'source':str(source),'sha256':sha(source),'rate':sr,'channels':ch,'family':family,'width':width,'targets':targets})
(out/'sources.json').write_text(json.dumps({'protocol_sha256':sha(out/'protocol.json'),'rows':rows},indent=2));print('Frozen',len(rows),'sources',sum(len(r['targets']) for r in rows),'conditions before rendering')

sources=rows
for label in ['baseline','candidate']:(out/label).mkdir(exist_ok=True)
def score(wav,ref):
 r=subprocess.run([str(judge),'--threads','1','--priority','below','--ep','cpu','check',str(wav),'--ref',str(ref),'--content','music','--json'],capture_output=True,encoding='utf-8',timeout=300);d=json.loads(r.stdout);assert 'detectors' in d;return d

def measure(y,t,sr,spans):
 mask=np.ones(len(t),dtype=bool);margin=round(.002*sr)
 for start,end in spans:mask[max(0,start-margin):min(len(t),end+margin)]=False
 energy=float(np.sum(t*t));ghost=float(10*np.log10(max(1e-30,np.sum(y[mask]**2)/max(1e-30,energy))));packets=[]
 for start,end in spans:
  if start<0 or end>len(t):continue
  target=t[start:end];te=float(np.sum(target*target));best=(-2,0,0)
  for lag in range(-round(.001*sr),round(.001*sr)+1):
   if start+lag<0 or end+lag>len(y):continue
   a=y[start+lag:end+lag];ae=float(np.sum(a*a));ncc=float(np.sum(a*target)/np.sqrt(max(1e-30,ae*te)))
   if ncc>best[0]:best=(ncc,lag,10*np.log10(max(1e-30,ae/max(1e-30,te))))
  packets.append({'ncc':best[0],'lag_samples':best[1],'energy_error_db':best[2]})
 return {'ghost_db':ghost,'packets':packets}
def one(job):
 row,ratio=job;key=f'{row["id"]:02d}_s{ratio:g}';target=row['targets'][str(ratio)];ref=Path(target['path']);record={};waveforms={};wav_paths={}
 for label,exe in [('baseline',baseline),('candidate',candidate)]:
  wav=out/label/(key+'.wav');subprocess.run([str(exe),row['source'],str(wav),'--multi','--tier','offline','--pitch','0','--stretch',str(ratio)],capture_output=True,check=True,timeout=180);wav_paths[label]=wav;waveforms[label],output_rate=sf.read(wav,always_2d=True);assert output_rate==row['rate']
  if label=='baseline' and target['reference_kind']=='golden_compatibility':ref.write_bytes(wav.read_bytes())
  if label=='candidate' and wav.read_bytes()==wav_paths['baseline'].read_bytes():d=json.loads((out/'baseline'/(key+'.json')).read_text())
  else:d=score(wav,ref)
  (out/label/(key+'.json')).write_text(json.dumps(d));record[label]={'scores':extract(d)}
 src,sr=sf.read(row['source'],always_2d=True);truth,rr=sf.read(ref,always_2d=True);assert sr==rr;old,new=waveforms['baseline'],waveforms['candidate'];reg=issues(record['baseline']['scores'],record['candidate']['scores']);physical=bool(np.isfinite(new).all() and new.shape==(int(np.floor(len(src)*ratio+.5)),src.shape[1]) and np.max(abs(new))<=max(np.max(abs(old)),np.max(abs(src))*np.sqrt(2))*(1+1e-5));identical=wav_paths['baseline'].read_bytes()==wav_paths['candidate'].read_bytes();exact_target=bool(np.array_equal(new,truth))
 if target['fallback_expected'] and not identical:reg.append(('fallback-changed',''))
 if not target['fallback_expected'] and not exact_target:reg.append(('target-not-exact',''))
 if not physical:reg.append(('physical',''))
 if target['reference_kind']=='known_events':
  for label,y in waveforms.items():record[label]['analytic']=measure(y,truth,sr,target['spans'])
  a=record['baseline']['analytic'];b=record['candidate']['analytic']
  if max(a['ghost_db'],b['ghost_db'])>-80 and b['ghost_db']>a['ghost_db']+1:reg.append(('ghost',''))
  for i,(x,y) in enumerate(zip(a['packets'],b['packets'])):
   if y['ncc']<x['ncc']-.01:reg.append(('packet-ncc',str(i)))
   if abs(y['energy_error_db'])>abs(x['energy_error_db'])+1:reg.append(('packet-energy-error',str(i)))
 record.update({'source_id':row['id'],'family':row['family'],'ratio':ratio,'reference_kind':target['reference_kind'],'fallback_expected':target['fallback_expected'],'identical_audio':identical,'exact_target':exact_target,'physical_pass':physical,'regressions':reg});return key,record
jobs=[(row,r) for row in sources for r in p['ratios']];rows={}
with ThreadPoolExecutor(args.workers) as pool:
 for i,(key,row) in enumerate(pool.map(one,jobs),1):
  rows[key]=row
  if i%30==0:print(i,len(jobs),flush=True)
known={k:r for k,r in rows.items() if r['reference_kind']=='known_events'};tt={s:totals({k:r[s]['scores'] for k,r in known.items()},list(known)) for s in ['baseline','candidate']};gain=float(np.mean([max(-80,r['baseline']['analytic']['ghost_db'])-max(-80,r['candidate']['analytic']['ghost_db']) for r in known.values()]));substantial=[k for k,r in known.items() if r['baseline']['analytic']['ghost_db']>=-50 and r['candidate']['analytic']['ghost_db']<=r['baseline']['analytic']['ghost_db']-6];reg={k:r['regressions'] for k,r in rows.items() if r['regressions']};quality=bool(np.isfinite(tt['candidate']['qual']) and tt['candidate']['qual']>=tt['baseline']['qual']+.002);a4=not(np.isfinite(tt['baseline']['qual']) and (not np.isfinite(tt['candidate']['qual']) or tt['candidate']['qual']<tt['baseline']['qual']-.002));report={'cases':len(rows),'known_event_cases':len(known),'fallback_cases':sum(r['fallback_expected'] for r in rows.values()),'exact_target_admitted':sum(not r['fallback_expected'] and r['exact_target'] for r in rows.values()),'fallback_exact':sum(r['fallback_expected'] and r['identical_audio'] for r in rows.values()),'regressions':reg,'totals_known_targets':tt,'ghost_gain_db':gain,'substantial_cases':substantial,'quality_gain_pass':quality,'a4':a4,'improvement':quality or gain>=3 and len(substantial)>=6,'rows':rows};report['pass']=not reg and a4 and report['improvement'];(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:len(v) if k in ['substantial_cases','regressions'] else v for k,v in report.items() if k!='rows'},indent=2));raise SystemExit(0 if report['pass'] else 1)
