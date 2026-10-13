#!/usr/bin/env python3
import struct, sys, json

path = sys.argv[1]
f = open(path, 'rb')

def rd(fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))

magic = f.read(4)
ver, n_tensors, n_kv = rd('<IQq') if False else (None,None,None)
# ggml gguf v3: magic(4) version(u32) n_tensors(u64) n_kv(u64)
f.seek(4)
ver = rd('<I')[0]
n_tensors = rd('<Q')[0]
n_kv = rd('<Q')[0]
print("magic", magic, "ver", ver, "n_tensors", n_tensors, "n_kv", n_kv)

def read_val(t):
    if t == 0: return rd('<B')[0]
    if t == 1: return rd('<b')[0]
    if t == 2: return rd('<H')[0]
    if t == 3: return rd('<h')[0]
    if t == 4: return rd('<I')[0]
    if t == 5: return rd('<i')[0]
    if t == 6: return rd('<f')[0]
    if t == 7: return rd('<?')[0]
    if t == 8:
        n = rd('<Q')[0]
        return f.read(n).decode('utf-8', 'replace')
    if t == 9:
        et = rd('<I')[0]
        n = rd('<Q')[0]
        return [read_val(et) for _ in range(n)]
    if t == 10: return rd('<Q')[0]
    if t == 11: return rd('<q')[0]
    if t == 12: return rd('<d')[0]
    raise ValueError('bad type %d' % t)

kvs = {}
for _ in range(n_kv):
    n = rd('<Q')[0]
    key = f.read(n).decode('utf-8', 'replace')
    t = rd('<I')[0]
    val = read_val(t)
    kvs[key] = val

want_sub = ('head', 'embd', 'block_count', 'context_length', 'rope', 'feed', 'arch', 'expert', 'layer', 'kv')
for k in sorted(kvs):
    if any(s in k for s in want_sub) or not k.startswith('tokenizer'):
        if k.startswith('tokenizer.'):
            continue
        print(f"{k} = {kvs[k]}")

print("\n--- tensors ---")
names = []
for _ in range(n_tensors):
    n = rd('<Q')[0]
    name = f.read(n).decode('utf-8', 'replace')
    n_dims = rd('<I')[0]
    dims = [rd('<Q')[0] for _ in range(n_dims)]
    t = rd('<I')[0]
    offs = rd('<Q')[0]
    names.append((name, dims, t))
print("total", len(names))
for name, dims, t in names:
    if 'attn' in name or 'blk.0.' in name:
        print(name, dims, 'type=', t)
