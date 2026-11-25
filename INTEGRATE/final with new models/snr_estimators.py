"""Estimate a clip's signal-to-noise ratio from its samples.

energy_details uses quiet frames to estimate the background. wada_details uses the
amplitude distribution and CMU's alpha=0.4 lookup table for Kim and Stern's WADA
method. The table and its license are in assets/wada/.
"""

import hashlib
import math
from functools import lru_cache
from pathlib import Path
import numpy as np

FRAME_SECONDS=.020
HOP_SECONDS=.010
METHODS=('pause_energy','wada_snr')
WADA_TABLE_SHA256='9afbf39a19f4c9159335120e188ca9bdd56947a513ce022c2e98d5f6ce3fc24c'


# Reject samples that can't give a meaningful estimate, including silence and constant
# offsets.
def valid_audio(samples,rate):
    x=np.asarray(samples,dtype=np.float64)
    if not isinstance(rate,(int,np.integer)) or rate<=0 or x.ndim!=1 or not np.isfinite(x).all():
        raise ValueError('Expected finite mono audio and a positive integer sample rate.')
    if len(x)<round(.1*rate):raise ValueError('Insufficient audio for SNR estimation.')
    if float(np.max(np.abs(x-x.mean())))<1e-12:raise ValueError('Silence/DC contains no usable speech/noise contrast.')
    return x


# Use active frames and measured quiet audio to estimate signal and background power.
def energy_details(samples,rate=16000,pause_intervals=None):
    result=dict(value_db=None,status='unavailable',noise_source='unavailable',noise_anchors=[],
                noise_variation_db=None,noise_frames=0,active_frames=0,notes=[])
    try:x=valid_audio(samples,rate)
    except ValueError as exc:result['notes']=[str(exc)];return result
    n,hop=round(rate*FRAME_SECONDS),round(rate*HOP_SECONDS)
    starts=np.arange(0,len(x)-n+1,hop);ends=starts+n
    raw=np.lib.stride_tricks.sliding_window_view(x,n)[::hop]
    # True digital-zero frames cannot supply an acoustic noise floor.
    # Long unresolved spans can contain hundreds of thousands of frames.
    # Bound temporaries while retaining the global mean and frame alignment.
    nonzero = np.empty(len(raw), dtype=bool)
    power = np.empty(len(raw))
    mean = x.mean()
    for first in range(0, len(raw), 2048):
        batch = raw[first:first+2048]
        nonzero[first:first+len(batch)] = np.any(batch != 0., axis=1)
        power[first:first+len(batch)] = np.mean((batch-mean)**2, axis=1)
    valid=nonzero & (power>1e-12)
    begin,end=starts/rate,ends/rate;centers=(begin+end)/2
    if pause_intervals is None:
        if valid.sum()<20:result['notes']=['Too little non-zero audio.'];return result
        pause=valid & (power<=np.percentile(power[valid],15))
        result['noise_source']='quiet_frame_fallback'
        result['notes'].append('No pause metadata: quietest-frame fallback is uncertain; review this estimate.')
    else:
        pause=np.zeros(len(power),dtype=bool)
        try:
            for a,b in pause_intervals:
                if not math.isfinite(float(a)+float(b)) or not 0<=a<b<=len(x)/rate+1/rate:
                    raise ValueError('Invalid clip-local pause interval.')
                pause|=(begin>=a-1e-9)&(end<=b+1e-9)
        except (TypeError,ValueError) as exc:result['notes']=[str(exc)];return result
        result['noise_source']='reviewed_part4_pause_context'
    noise_mask=pause & valid;active_mask=(~pause)&valid
    result.update(noise_frames=int(noise_mask.sum()),active_frames=int(active_mask.sum()))
    if noise_mask.sum()<10 or active_mask.sum()<10:
        result['notes'].append('Insufficient non-zero pause or active frames; no reliable energy SNR.')
        return result
    # Independent contiguous pause runs become robust noise anchors. Long runs
    # are binned into 0.25 s anchors so observed changes can be followed locally.
    indexes=np.flatnonzero(noise_mask)
    runs=np.split(indexes,np.flatnonzero(np.diff(indexes)>1)+1)
    anchors=[]
    for run in runs:
        for part in np.array_split(run,max(1,math.ceil(len(run)*HOP_SECONDS/.25))):
            if len(part)>=3:anchors.append((float(np.mean(centers[part])),float(np.mean(power[part]))))
    if not anchors:result['notes'].append('No stable noise anchors.');return result
    times=np.array([a for a,b in anchors]);noise=np.array([b for a,b in anchors])
    profile=np.interp(centers,times,noise)
    observed=float(np.mean(power[active_mask]));floor=float(np.mean(profile[active_mask]));signal=observed-floor
    result['noise_anchors']=[dict(time_seconds=a,power=b,level_dbfs=10*math.log10(b)) for a,b in anchors]
    variation=float(10*np.log10(noise.max()/noise.min()))
    result.update(noise_variation_db=round(variation,2),observed_power=observed,noise_power=floor,
                  signal_power=signal,noise_tracking='linear power interpolation between measured pause anchors; constant beyond outer anchors')
    if signal<=0 or floor<=0:
        result['notes'].append('Active energy does not exceed measured background noise.');return result
    value=10*math.log10(signal/floor)
    result.update(value_db=round(value,2),status='ok')
    if variation>6:result['notes'].append('Background noise varies by more than 6 dB; interpolation between pauses is uncertain.')
    if observed/floor<1.25:result['notes'].append('Very low speech/noise contrast; subtraction is sensitive to noise-estimation error.')
    return result


# Read the lookup table that converts WADA's amplitude statistic into decibels.
@lru_cache(maxsize=1)
def wada_table():
    path=Path(__file__).resolve().parent/'assets/wada/Alpha0.400000.txt'
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=WADA_TABLE_SHA256:raise ValueError('CMU WADA table checksum mismatch; restore the packaged reference table.')
    rows=[(float(line.split()[0]),float(line.split()[-1])) for line in raw.decode('ascii').splitlines() if line.strip()]
    table=np.asarray(rows,dtype=np.float64)
    if table.shape!=(121,2) or not np.array_equal(table[:,0],np.arange(-20,101)) or not np.isfinite(table).all():
        raise ValueError('Invalid CMU WADA alpha=0.4 table.')
    return table,hashlib.sha256(raw).hexdigest()


# Measure the amplitude distribution and flag estimates that hit the limits of the table.
def wada_details(samples,rate=16000):
    result=dict(value_db=None,status='unavailable',bound=None,blocks=[],notes=[])
    try:
        x=valid_audio(samples,rate);table,checksum=wada_table()
    except (ValueError,OSError) as exc:result['notes']=[str(exc)];return result
    # Remove whole digital-zero 20 ms blocks, not low-energy natural noise.
    # Isolated zero crossings are handled by the published log floor below.
    size=max(1,round(.02*rate));keep=np.ones(len(x),dtype=bool)
    for a in range(0,len(x),size):
        if not np.any(x[a:a+size]):keep[a:a+size]=False
    removed=int((~keep).sum());x=x[keep]
    if len(x)<round(.5*rate):result['notes']=['Too little non-zero audio for WADA.'];return result
    if removed:result['notes'].append(f'{removed/rate:.3f} s of whole digital-zero blocks excluded; zeros are not acoustic-noise measurements.')
    signal_energy=noise_energy=0.;bounds=[]
    # Balanced <=100000-sample blocks avoid silently dropping a short tail.
    for block in np.array_split(x,max(1,math.ceil(len(x)/100000))):
        block=block-block.mean();peak=float(np.max(np.abs(block)))
        if peak<1e-12:continue
        normalized=block/peak;absolute=np.maximum(np.abs(normalized),1e-10)
        statistic=float(np.log(np.mean(absolute))-np.mean(np.log(absolute)))
        # Match CMU's first-threshold-above-G lookup, including its slightly
        # non-monotone Monte Carlo values at the low end; do not sort the table.
        found=np.flatnonzero(statistic<table[:,1])
        index=int(found[0]) if len(found) else len(table)-1
        score=float(table[index,0]);bound='below_range' if index==0 else ('above_range' if not len(found) else None)
        ratio=10**(score/10);energy=float(np.sum(block**2))
        noise_energy+=energy/(1+ratio);signal_energy+=energy*ratio/(1+ratio)
        result['blocks'].append(dict(samples=len(block),g_statistic=statistic,snr_db=score,bound=bound))
        if bound:bounds.append(bound)
    if noise_energy<=0 or signal_energy<=0:result['notes'].append('No usable WADA block.');return result
    value=10*math.log10(signal_energy/noise_energy)
    result.update(value_db=round(value,2),status='ok',table_sha256=checksum,alpha=.4,table_range_db=[-20,100])
    if bounds:
        result['bound']=bounds[0] if len(set(bounds))==1 else 'mixed_bounds'
        result['status']='range_limited'
        result['notes'].append('At least one block reaches the reference-table limit; value is not an exact in-range measurement.')
    result['notes'].append('Blind speech-distribution estimate (Gamma alpha=0.4, Gaussian-noise model); not a clean-reference measurement.')
    return result
