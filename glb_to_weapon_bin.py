#!/usr/bin/env python3
"""
Convert a Source2Viewer weapon GLB into a rigid weapon chams .bin format.

All vertices are baked into bone 0's local space (using inverse bind matrices)
so the runtime only needs a single world transform (bone 0 pos + quat).

Output format:
  [u32 vc][u32 ic][u32 bc=1][vert_data: vc × 56][index_data: ic × u16][pad if odd][ibm_data: 1 × 48]

Vertex layout (56 bytes): 3f pos + 3f norm + 4f bone_indices + 4f bone_weights
All vertices have bone_index=0 and weight=1.0 (rigid).
"""
import struct
import json
import argparse
import os
import math


def parse_glb(path):
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:4] == b'glTF', f"Not a GLB file: {path}"
    json_chunk_len = struct.unpack('<I', data[12:16])[0]
    gltf = json.loads(data[20:20 + json_chunk_len])
    bin_offset = 20 + json_chunk_len
    bin_chunk_len = struct.unpack('<I', data[bin_offset:bin_offset+4])[0]
    bin_data = data[bin_offset+8:bin_offset+8+bin_chunk_len]
    return gltf, bin_data


def get_accessor(gltf, bin_data, idx):
    acc = gltf['accessors'][idx]
    bv = gltf['bufferViews'][acc['bufferView']]
    offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    count = acc['count']
    ct = acc['componentType']; ts = acc['type']
    cs = {5120:1,5121:1,5122:2,5123:2,5125:4,5126:4}[ct]
    cf = {5120:'b',5121:'B',5122:'h',5123:'H',5125:'I',5126:'f'}[ct]
    tc = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[ts]
    stride = bv.get('byteStride', cs * tc)
    result = []
    for i in range(count):
        item_off = offset + i * stride
        elem = []
        for j in range(tc):
            elem.append(struct.unpack_from('<' + cf, bin_data, item_off + j*cs)[0])
        result.append(elem if tc > 1 else elem[0])
    return result, ts, ct


def colmajor_to_rowmajor(m_cm):
    return [[m_cm[c*4+r] for c in range(4)] for r in range(4)]


def mat_mul(A, B):
    R = [[0.0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            R[i][j] = sum(A[i][k]*B[k][j] for k in range(4))
    return R


def mat_inv(M):
    m = [row[:] for row in M]
    inv = [[1.0 if i==j else 0.0 for j in range(4)] for i in range(4)]
    for col in range(4):
        max_r = col; max_v = abs(m[col][col])
        for r in range(col+1, 4):
            if abs(m[r][col]) > max_v:
                max_r, max_v = r, abs(m[r][col])
        if max_v < 1e-12:
            raise RuntimeError("Singular matrix")
        if max_r != col:
            m[col], m[max_r] = m[max_r], m[col]
            inv[col], inv[max_r] = inv[max_r], inv[col]
        pv = m[col][col]
        for j in range(4):
            m[col][j] /= pv; inv[col][j] /= pv
        for r in range(4):
            if r != col:
                f = m[r][col]
                for j in range(4):
                    m[r][j] -= f*m[col][j]; inv[r][j] -= f*inv[col][j]
    return inv


def transform_point(M, p):
    """Apply 4x4 matrix to a 3D point (w=1)."""
    return [
        M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]*p[2] + M[0][3],
        M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]*p[2] + M[1][3],
        M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]*p[2] + M[2][3],
    ]


def transform_normal(M, n):
    """Apply 3x3 part of matrix to a normal."""
    return [
        M[0][0]*n[0] + M[0][1]*n[1] + M[0][2]*n[2],
        M[1][0]*n[0] + M[1][1]*n[1] + M[1][2]*n[2],
        M[2][0]*n[0] + M[2][1]*n[1] + M[2][2]*n[2],
    ]


def normalize(v):
    L = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if L < 1e-9:
        return v
    return [v[0]/L, v[1]/L, v[2]/L]


def node_local(node):
    if 'matrix' in node:
        return colmajor_to_rowmajor(node['matrix'])
    tr = node.get('translation', [0,0,0])
    rt = node.get('rotation', [0,0,0,1])
    sc = node.get('scale', [1,1,1])
    T = [[1,0,0,tr[0]],[0,1,0,tr[1]],[0,0,1,tr[2]],[0,0,0,1]]
    x,y,z,w = rt
    xx,yy,zz = x*x,y*y,z*z; xy,xz,yz = x*y,x*z,y*z; wx,wy,wz = w*x,w*y,w*z
    R = [[1-2*(yy+zz),2*(xy-wz),2*(xz+wy),0],
         [2*(xy+wz),1-2*(xx+zz),2*(yz-wx),0],
         [2*(xz-wy),2*(yz+wx),1-2*(xx+yy),0],
         [0,0,0,1]]
    S = [[sc[0],0,0,0],[0,sc[1],0,0],[0,0,sc[2],0],[0,0,0,1]]
    return mat_mul(T, mat_mul(R, S))


def build_parent_map(gltf):
    parent = {}
    for i, n in enumerate(gltf['nodes']):
        for c in n.get('children', []):
            parent[c] = i
    return parent


def joint_world(gltf, parent_map, node_idx):
    chain = []
    cur = node_idx
    while cur is not None:
        chain.append(cur); cur = parent_map.get(cur)
    chain.reverse()
    W = [[1.0 if i==j else 0.0 for j in range(4)] for i in range(4)]
    for n_idx in chain:
        W = mat_mul(W, node_local(gltf['nodes'][n_idx]))
    return W


def find_best_mesh(gltf):
    """Find the best mesh — prefer body_hd (high detail), fall back to body_legacy."""
    # Try HD first
    for i, m in enumerate(gltf.get('meshes', [])):
        name = m.get('name', '').lower()
        if '_hd' in name or 'body_hd' in name:
            for n in gltf.get('nodes', []):
                if n.get('mesh') == i and 'skin' in n:
                    print(f"  Using HD mesh: {m.get('name','?')}")
                    return i, n['skin']
    # Fall back to legacy
    for i, m in enumerate(gltf.get('meshes', [])):
        name = m.get('name', '').lower()
        if 'legacy' in name:
            for n in gltf.get('nodes', []):
                if n.get('mesh') == i and 'skin' in n:
                    print(f"  Using legacy mesh: {m.get('name','?')}")
                    return i, n['skin']
    # Last resort: first skinned mesh
    for i, m in enumerate(gltf.get('meshes', [])):
        for n in gltf.get('nodes', []):
            if n.get('mesh') == i and 'skin' in n:
                print(f"  Using first skinned mesh: {m.get('name','?')}")
                return i, n['skin']
    raise RuntimeError("No skinned mesh found")


def get_glb_ibms(gltf, bin_data, skin):
    """Read all inverse bind matrices from the GLB."""
    if 'inverseBindMatrices' not in skin:
        return None
    acc = gltf['accessors'][skin['inverseBindMatrices']]
    bv = gltf['bufferViews'][acc['bufferView']]
    offset = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    count = acc['count']
    ibms = []
    for i in range(count):
        raw = struct.unpack('<16f', bin_data[offset + i*64 : offset + i*64 + 64])
        ibms.append(colmajor_to_rowmajor(list(raw)))
    return ibms


def convert(glb_path, bin_path):
    gltf, bin_data = parse_glb(glb_path)
    mesh_idx, skin_idx = find_best_mesh(gltf)
    mesh = gltf['meshes'][mesh_idx]
    skin = gltf['skins'][skin_idx]
    parent_map = build_parent_map(gltf)

    bc_orig = len(skin['joints'])
    bone_names = [gltf['nodes'][jn].get('name', f'joint_{ji}') for ji, jn in enumerate(skin['joints'])]
    print(f"  Original bones ({bc_orig}): {bone_names}")

    # Get GLB inverse bind matrices
    glb_ibms = get_glb_ibms(gltf, bin_data, skin)
    if glb_ibms is None:
        print("  ERROR: No inverse bind matrices in GLB")
        return

    # Bone 0 IBM and its inverse (= bind matrix of bone 0)
    ibm0 = glb_ibms[0]
    bind0 = mat_inv(ibm0)  # bind matrix = world position of bone 0 in mesh space

    # For each bone, compute: transform_to_bone0 = ibm0 * bind_i
    # This takes a vertex from bone_i's local space into bone 0's local space
    # But actually, we want: for a vertex weighted to bone_i,
    #   mesh_pos -> ibm_i -> bone_i_local -> bind_i -> mesh_pos -> ibm0 -> bone0_local
    # Simplified: bone0_local_pos = ibm0 * inverse(ibm_i) * ibm_i * mesh_pos = ibm0 * mesh_pos
    # Wait, that's just ibm0 applied to all vertices regardless of bone weight!
    # Because all bones share the same mesh space, we just need ibm0 to go to bone 0 local.

    # === VERTEX PROCESSING ===
    # Apply IBM0 to ALL vertices to put them in bone 0's local space
    new_pos, new_norm, new_indices = [], [], []
    dedup_map = {}
    dedup_hits = 0

    def vert_key(p, n):
        return (
            round(p[0]*10000), round(p[1]*10000), round(p[2]*10000),
            round(n[0]*1000), round(n[1]*1000), round(n[2]*1000),
        )

    for prim in mesh['primitives']:
        attrs = prim['attributes']
        pos, _, _ = get_accessor(gltf, bin_data, attrs['POSITION'])
        norm, _, _ = get_accessor(gltf, bin_data, attrs.get('NORMAL', attrs['POSITION']))

        if 'indices' in prim:
            indices, _, _ = get_accessor(gltf, bin_data, prim['indices'])
        else:
            indices = list(range(len(pos)))

        local_to_new = []
        for vi in range(len(pos)):
            # Transform vertex position and normal by IBM0
            tp = transform_point(ibm0, pos[vi])
            tn = normalize(transform_normal(ibm0, norm[vi]))

            key = vert_key(tp, tn)
            existing = dedup_map.get(key)
            if existing is not None:
                local_to_new.append(existing)
                dedup_hits += 1
                continue

            new_idx = len(new_pos)
            dedup_map[key] = new_idx
            local_to_new.append(new_idx)
            new_pos.append(tp)
            new_norm.append(tn)

        for ti in range(0, len(indices), 3):
            a, b, c = indices[ti], indices[ti+1], indices[ti+2]
            na, nb, nc = local_to_new[a], local_to_new[b], local_to_new[c]
            new_indices.extend([na, nb, nc])

    if dedup_hits > 0:
        print(f"  Deduplicated {dedup_hits} vertices")

    vc = len(new_pos)
    ic = len(new_indices)
    bc = 1  # Single rigid bone
    use_32bit = vc > 65535

    print(f"  Final: {vc} verts, {ic} indices, {bc} bone (rigid){' [32-bit idx]' if use_32bit else ''}")

    # === WRITE OUTPUT ===
    # Header: [u32 vc][u32 ic][u32 bc]
    # bc high bit (bit 31) = 1 means 32-bit indices
    bc_flags = bc | (0x80000000 if use_32bit else 0)
    out = bytearray()
    out += struct.pack('<III', vc, ic, bc_flags)

    for i in range(vc):
        p = new_pos[i]; n = new_norm[i]
        out += struct.pack('<fff', p[0], p[1], p[2])           # position (in bone 0 local space)
        out += struct.pack('<fff', n[0], n[1], n[2])           # normal
        out += struct.pack('<ffff', 0.0, 0.0, 0.0, 0.0)       # bone indices (all bone 0)
        out += struct.pack('<ffff', 1.0, 0.0, 0.0, 0.0)       # weights (100% bone 0)

    if use_32bit:
        for idx in new_indices:
            out += struct.pack('<I', idx)
    else:
        for idx in new_indices:
            out += struct.pack('<H', idx)
        if ic % 2 != 0:
            out += b'\x00\x00'

    # === IBM for bone 0: identity (vertices are already in bone 0 local space) ===
    for r in range(3):
        for c in range(4):
            out += struct.pack('<f', 1.0 if r == c else 0.0)

    with open(bin_path, 'wb') as f:
        f.write(bytes(out))
    print(f"  Wrote {bin_path}: {len(out)} bytes\n")


def batch_convert(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    glbs = [f for f in os.listdir(input_dir) if f.endswith('.glb')]
    if not glbs:
        print(f"No .glb files found in {input_dir}")
        return
    for glb in sorted(glbs):
        name = os.path.splitext(glb)[0]
        bin_name = name + ".bin"
        print(f"=== Converting {glb} ===")
        try:
            convert(os.path.join(input_dir, glb), os.path.join(output_dir, bin_name))
        except Exception as e:
            print(f"  ERROR: {e}\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Convert weapon GLBs to rigid chams .bin format")
    ap.add_argument('input', help="GLB file or directory of GLBs")
    ap.add_argument('output', help="Output .bin file or directory")
    ap.add_argument('--batch', action='store_true', help="Convert all GLBs in input dir")
    args = ap.parse_args()

    if args.batch:
        batch_convert(args.input, args.output)
    else:
        convert(args.input, args.output)
