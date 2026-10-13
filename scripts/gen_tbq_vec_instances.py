#!/usr/bin/env python3
"""Generate TurboQuant (TBQ/TBQP) fattn-vec template instances for prism-ml."""
import os

D = os.path.join(os.path.dirname(__file__),
                 '..', 'llama.cpp-prism-ml', 'ggml', 'src', 'ggml-cuda', 'template-instances')

# (head_dim, type_K, type_V)
combos = []

for s in ('0', '1'):  # _0 = 256-block (D=256), _1 = 128-block (D=128)
    d = 256 if s == '0' else 128
    combos += [
        (d, 'f16',     f'tbq3_{s}'),
        (d, 'f16',     f'tbq4_{s}'),
        (d, f'tbq3_{s}',  'f16'),
        (d, f'tbq4_{s}',  'f16'),
        (d, f'tbqp3_{s}', 'f16'),
        (d, f'tbqp4_{s}', 'f16'),
        (d, f'tbq3_{s}',  f'tbq3_{s}'),
        (d, f'tbq3_{s}',  f'tbq4_{s}'),
        (d, f'tbq4_{s}',  f'tbq3_{s}'),
        (d, f'tbq4_{s}',  f'tbq4_{s}'),
        (d, f'tbqp3_{s}', f'tbq3_{s}'),
        (d, f'tbqp3_{s}', f'tbq4_{s}'),
        (d, f'tbqp4_{s}', f'tbq3_{s}'),
        (d, f'tbqp4_{s}', f'tbq4_{s}'),
    ]

# D=64: K uses the double-WHT _3 family, V uses the _2 family
combos += [
    (64, 'f16',     'tbq3_2'),
    (64, 'f16',     'tbq4_2'),
    (64, 'tbq3_2',  'f16'),
    (64, 'tbq3_3',  'f16'),
    (64, 'tbq4_3',  'f16'),
    (64, 'tbqp3_3', 'f16'),
    (64, 'tbqp4_3', 'f16'),
    (64, 'tbq3_3',  'tbq3_2'),
    (64, 'tbq3_3',  'tbq4_2'),
    (64, 'tbq4_3',  'tbq3_2'),
    (64, 'tbq4_3',  'tbq4_2'),
    (64, 'tbqp3_3', 'tbq3_2'),
    (64, 'tbqp3_3', 'tbq4_2'),
    (64, 'tbqp4_3', 'tbq3_2'),
    (64, 'tbqp4_3', 'tbq4_2'),
]

PREAMBLE_TBQ_V = """// TurboQuant KV cache instance (ported from llama.cpp-AmesianX-TurboQuant).
// Force float accumulation for TBQ V types
#include "../common.cuh"
#undef V_DOT2_F32_F16_AVAILABLE

#include "../fattn-vec.cuh"
"""

PREAMBLE = """// TurboQuant KV cache instance (ported from llama.cpp-AmesianX-TurboQuant).
#include "../fattn-vec.cuh"
"""

count = 0
for d, tk, tv in combos:
    is_tbq_v = tv.startswith('tbq') or tv.startswith('amx')
    body = PREAMBLE_TBQ_V if is_tbq_v else PREAMBLE
    body += f"\nDECL_FATTN_VEC_CASE({d:3d}, GGML_TYPE_{tk.upper()}, GGML_TYPE_{tv.upper()});\n"
    path = os.path.join(D, f"fattn-vec-instance-{tk}-{tv}.cu")
    if os.path.exists(path):
        print("skip (exists)", os.path.basename(path))
        continue
    with open(path, 'w') as f:
        f.write(body)
    count += 1
    print("wrote", os.path.basename(path))
print("total written:", count)
