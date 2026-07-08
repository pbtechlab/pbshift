# Doubler/chorus as SPECTRAL SIDEBANDS: a clean stretched tone keeps razor-sharp
# harmonic lines; chorus/doubler adds close sidebands (a detuned copy). Measures
# median sideband-to-carrier ratio (dB) over harmonics of a steady tone. Lower=cleaner.
import sys, numpy as np, soundfile as sf
x,sr=sf.read(sys.argv[1])
if x.ndim>1: x=x.mean(1)
n=len(x); x=x[int(0.25*n):int(0.75*n)]
w=np.hanning(len(x)); X=np.abs(np.fft.rfft(x*w)); f=np.fft.rfftfreq(len(x),1/sr)
def bin(hz): return int(round(hz*len(x)/sr))
f0=220.0
sc=[]
for k in range(1,9):
    c=bin(f0*k); 
    if c+bin(80)>=len(X): break
    peak=X[max(0,c-bin(3)):c+bin(3)].max()
    # sidebands: 10..70 Hz either side, excluding mainlobe
    sb=np.concatenate([X[c-bin(70):c-bin(10)], X[c+bin(10):c+bin(70)]])
    if len(sb)==0 or peak<=0: continue
    sc.append(20*np.log10(sb.max()/peak+1e-9))
print(f"  sideband/carrier={np.median(sc):+6.1f}dB  (lower=cleaner, per-harmonic median over {len(sc)} harmonics)")
