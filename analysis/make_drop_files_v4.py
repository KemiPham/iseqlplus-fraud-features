"""Writes data4/train_drop{K}.parquet: the training split without the legitimate row listed in data4/drop_manifest.json (created by train_v4.py)."""
import json, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
drops=json.load(open('data4/drop_manifest.json'))
for k,ids in drops.items():
    T=pq.read_table('data4/train.parquet'); T=T.filter(pc.invert(pc.is_in(T['TransactionID'],pa.array(ids))))
    pq.write_table(T,f'data4/train_drop{k}.parquet',row_group_size=50000); print(k,ids,T.num_rows,flush=True); del T
