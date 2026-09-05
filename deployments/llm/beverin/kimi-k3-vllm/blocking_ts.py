import json, sys
from collections import defaultdict
path = sys.argv[1]
with open(path) as fh: data = json.load(fh)
evs = data.get("traceEvents", data) if isinstance(data, dict) else data
gpu = [e for e in evs if e.get("cat")=="kernel" and "dur" in e]
st=defaultdict(float)
for e in gpu: st[(e.get("pid"),e.get("tid"))]+=e["dur"]
main,_=max(st.items(),key=lambda x:x[1])
nccl=[e for e in gpu if (e.get("pid"),e.get("tid"))==main and "nccl" in e.get("name","").lower()]
if nccl:
    ts_lo=min(e["ts"] for e in nccl); ts_hi=max(e["ts"]+e["dur"] for e in nccl)
    span=ts_hi-ts_lo
    blk=[e for e in nccl if e["dur"]>100000]
    print(f"NCCL span: {span/1e6:.2f}s, {len(nccl)} kern, {len(blk)} >100ms")
    print(f"blocking kernel timestamps (as % of span):")
    for e in sorted(blk,key=lambda x:x["ts"])[:30]:
        print(f"  {100*(e['ts']-ts_lo)/span:5.1f}%  dur={e['dur']/1e6:.3f}s")
