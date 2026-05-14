#!/usr/bin/env python3
"""
Convert a Source2Viewer GLB into the chams .bin format with name-based bone mapping.

Reverse-engineered from ctm_sas_mesh.bin: CS2's runtime bone array uses a SPECIFIC ORDER
of bones by name. We map GLB joints → CS2 bone slots by matching joint names.

Output format (matches load_chams_mesh_data):
  [u32 vc][u32 ic][u32 bc][vert_data: vc × 56][index_data: ic × u16][pad if odd][ibm_data: bc × 48]
"""
import struct
import json
import argparse


# === CS2 CT BONE ORDER (from reverse engineering ctm_sas_mesh.bin) ===
# Index → bone name (from GLB). Index 0 is "root_motion" (synthetic anchor).
CS2_CT_BONES = [
    "root_motion",        # 0  — synthetic, IBM is M_root rotation
    "pelvis",             # 1
    "spine_0",            # 2
    "spine_1",            # 3
    "spine_2",            # 4  (mapped to chesthier_offset in some agents — same position)
    "spine_3",            # 5
    "neck_0",             # 6
    "head_0",             # 7
    "clavicle_l",         # 8
    "arm_upper_l",        # 9
    "arm_lower_l",        # 10
    "hand_l",             # 11
    "clavicle_r",         # 12
    "arm_upper_r",        # 13
    "arm_lower_r",        # 14
    "hand_r",             # 15
    "jiggle_primary",     # 16
    "leg_upper_l",        # 17
    "leg_lower_l",        # 18
    "ankle_l",            # 19
    "leg_upper_r",        # 20
    "leg_lower_r",        # 21
    "ankle_r",            # 22
    "weapon_hand",        # 23 — typically maps to spine_3 or similar
    "root_motion",        # 24
    "eyeball_l",          # 25
    "eyeball_r",          # 26
    "eye_target",         # 27
    "jiggle_hood",        # 28
    "finger_middle_meta_l", # 29
    "finger_middle_0_l",  # 30
    "finger_middle_1_l",  # 31
    "finger_middle_2_l",  # 32
    "finger_meta_l",      # 33  (skipped in sas — placeholder)
    "finger_pinky_0_l",   # 34
    "finger_pinky_1_l",   # 35
    "finger_pinky_2_l",   # 36
    "finger_index_meta_l",# 37
    "finger_index_0_l",   # 38
    "finger_index_1_l",   # 39
    "finger_index_2_l",   # 40
    "finger_thumb_0_l",   # 41
    "finger_thumb_1_l",   # 42
    "finger_thumb_2_l",   # 43
    "finger_ring_meta_l", # 44
    "finger_ring_0_l",    # 45
    "finger_ring_1_l",    # 46
    "finger_ring_2_l",    # 47
    "arm_lower_l_twist",  # 48
    "arm_lower_l_twist1", # 49
    "arm_upper_l_twist",  # 50
    "arm_upper_l_twist1", # 51
    "scapula_l",          # 52
    "finger_middle_meta_r", # 53
    "finger_middle_0_r",  # 54
    "finger_middle_1_r",  # 55
    "finger_middle_2_r",  # 56
    "finger_meta_r",      # 57
    "finger_pinky_0_r",   # 58
    "finger_pinky_1_r",   # 59
    "finger_pinky_2_r",   # 60
    "finger_index_meta_r",# 61
    "finger_index_0_r",   # 62
    "finger_index_1_r",   # 63
    "finger_index_2_r",   # 64
    "finger_thumb_0_r",   # 65
    "finger_thumb_1_r",   # 66
    "finger_thumb_2_r",   # 67
    "finger_ring_meta_r", # 68
    "finger_ring_0_r",    # 69
    "finger_ring_1_r",    # 70
    "finger_ring_2_r",    # 71
    "arm_lower_r_twist",  # 72
    "arm_lower_r_twist1", # 73
    "arm_upper_r_twist",  # 74
    "arm_upper_r_twist1", # 75
    "scapula_r",          # 76
    "jiggle_front_micropouches", # 77
    "jiggle_radio",       # 78
    "jiggle_front_pouch_01", # 79
    "jiggle_front_pouch_02", # 80
    "ball_l",             # 81
    "leg_upper_l_twist",  # 82
    "leg_upper_l_twist1", # 83
    "jiggle_climbinggear_01", # 84
    "jiggle_climbinggear_02", # 85
    "ball_r",             # 86
    "leg_upper_r_twist",  # 87
    "leg_upper_r_twist1", # 88
    "jiggle_holster",     # 89
    "head_0_twist",       # 90
    "scap_aimup",         # 91
    "scap_r_aimat",       # 92
    "scap_l_aimat",       # 93
]

# Aliases — GLB might use slightly different names for same bone
ALIASES = {
    "head": "head_0",
    "ankle_l": "leg_l_iktarget",  # observed in sas
    "ankle_r": "leg_r_iktarget",
    "hand_l": "weaponhier_l_iktarget",  # observed in sas
    "hand_r": "weaponhier_r_iktarget",
    "spine_2": "chesthier_offset",  # observed in sas
    "weapon_hand": "weaponhier_jnt",
}


def parse_glb(path):
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:4] == b'glTF'
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


def find_body_mesh(gltf):
    cands = []
    for i, m in enumerate(gltf.get('meshes', [])):
        skin = None
        for n in gltf.get('nodes', []):
            if n.get('mesh') == i and 'skin' in n:
                skin = n['skin']; break
        if skin is None: continue
        vc = sum(gltf['accessors'][p['attributes']['POSITION']]['count']
                 for p in m['primitives'] if 'POSITION' in p['attributes'])
        cands.append((i, skin, m.get('name','').lower(), vc))
    for c in cands:
        if 'body' in c[2]:
            print(f"  Body mesh: {c[2]} ({c[3]} verts)")
            return c[0], c[1]
    if cands:
        biggest = max(cands, key=lambda x: x[3])
        print(f"  Largest mesh: {biggest[2]} ({biggest[3]} verts)")
        return biggest[0], biggest[1]
    raise RuntimeError("No skinned meshes")


def build_parent_map(gltf):
    parent = {}
    for i, n in enumerate(gltf['nodes']):
        for c in n.get('children', []):
            parent[c] = i
    return parent


def get_skeleton_root(gltf, skin):
    parent = build_parent_map(gltf)
    return parent.get(skin['joints'][0])


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


def build_glb_joint_lookup(gltf, skin):
    """name → (skin_index, node_index)"""
    lookup = {}
    for i, jn in enumerate(skin['joints']):
        name = gltf['nodes'][jn].get('name', '')
        lookup[name] = (i, jn)
    return lookup


def convert(glb_path, bin_path, max_bones, bone_table=None):
    if bone_table is None:
        bone_table = CS2_CT_BONES
    
    gltf, bin_data = parse_glb(glb_path)
    mesh_idx, skin_idx = find_body_mesh(gltf)
    mesh = gltf['meshes'][mesh_idx]
    skin = gltf['skins'][skin_idx]
    parent_map = build_parent_map(gltf)
    
    skel_root = get_skeleton_root(gltf, skin)
    M_root = joint_world(gltf, parent_map, skel_root) if skel_root is not None else \
             [[1.0 if i==j else 0.0 for j in range(4)] for i in range(4)]
    M_root_inv = mat_inv(M_root)
    
    # Build case-insensitive lookup
    glb_joint_by_name = {}
    for i, jn in enumerate(skin['joints']):
        name = gltf['nodes'][jn].get('name', '')
        glb_joint_by_name[name.lower()] = (i, jn)
    
    # === STRATEGY DETECTION ===
    # If GLB skin joint order already matches CS2 (e.g. joint[0] = root_motion, joint[1] = pelvis),
    # use direct index mapping. Otherwise fall back to name-based mapping.
    direct_mapping = True
    test_indices = [(0, 'root_motion'), (1, 'pelvis'), (7, 'head_0')]
    for idx, expected in test_indices:
        if idx >= len(skin['joints']):
            direct_mapping = False; break
        actual = gltf['nodes'][skin['joints'][idx]].get('name', '').lower()
        if actual != expected.lower():
            direct_mapping = False; break
    
    cs2_to_glb_node = [None] * max_bones
    cs2_to_glb_skin_idx = [None] * max_bones
    matched, unmatched = 0, []
    
    if direct_mapping:
        print(f"  Direct index mapping (GLB joint order = CS2 bone order)")
        # Just use skin index = CS2 bone index for as many as available
        for cs2_idx in range(min(max_bones, len(skin['joints']))):
            cs2_to_glb_skin_idx[cs2_idx] = cs2_idx
            cs2_to_glb_node[cs2_idx] = skin['joints'][cs2_idx]
            matched += 1
        for cs2_idx in range(len(skin['joints']), max_bones):
            unmatched.append((cs2_idx, f"joint_{cs2_idx}"))
    else:
        print(f"  Name-based mapping (GLB has nonstandard joint order)")
        for cs2_idx in range(min(max_bones, len(bone_table))):
            name = bone_table[cs2_idx].lower()
            if name in glb_joint_by_name:
                glb_skin_i, glb_node_i = glb_joint_by_name[name]
            elif name in ALIASES and ALIASES[name].lower() in glb_joint_by_name:
                glb_skin_i, glb_node_i = glb_joint_by_name[ALIASES[name].lower()]
            else:
                unmatched.append((cs2_idx, name)); continue
            cs2_to_glb_node[cs2_idx] = glb_node_i
            cs2_to_glb_skin_idx[cs2_idx] = glb_skin_i
            matched += 1
    
    print(f"Mesh: {mesh.get('name','?')} ({len(mesh['primitives'])} primitives)")
    print(f"Skin: {len(skin['joints'])} GLB joints, target {max_bones} CS2 bones")
    print(f"Matched: {matched}/{max_bones}")
    if unmatched:
        print(f"Unmatched CS2 slots ({len(unmatched)}): {unmatched[:10]}{'...' if len(unmatched)>10 else ''}")
    
    # Build reverse map: GLB skin index → CS2 bone slot
    glb_skin_to_cs2 = {}
    for cs2_idx, glb_si in enumerate(cs2_to_glb_skin_idx):
        if glb_si is not None:
            glb_skin_to_cs2[glb_si] = cs2_idx
    
    # === VERTEX PROCESSING ===
    new_pos, new_norm, new_joints, new_weights, new_indices = [], [], [], [], []
    dropped_v, dropped_t = 0, 0
    
    # Dedup: hash each unique vertex (pos+norm+joints+weights) → output index.
    # Quantize floats to ~6 decimal places via int rounding to handle FP noise.
    dedup_map = {}
    dedup_hits = 0
    
    def vert_key(p, n, j, w):
        # Quantize: pos to nearest 0.0001 inch, norm to 0.001, weights to 0.001
        return (
            round(p[0] * 10000), round(p[1] * 10000), round(p[2] * 10000),
            round(n[0] * 1000), round(n[1] * 1000), round(n[2] * 1000),
            j[0], j[1], j[2], j[3],
            round(w[0] * 1000), round(w[1] * 1000), round(w[2] * 1000), round(w[3] * 1000),
        )
    
    for prim in mesh['primitives']:
        attrs = prim['attributes']
        pos, _, _ = get_accessor(gltf, bin_data, attrs['POSITION'])
        norm, _, _ = get_accessor(gltf, bin_data, attrs.get('NORMAL', attrs['POSITION']))
        joints, _, _ = get_accessor(gltf, bin_data, attrs['JOINTS_0'])
        weights, _, ctype = get_accessor(gltf, bin_data, attrs['WEIGHTS_0'])
        if ctype == 5121:
            weights = [[w/255.0 for w in wv] for wv in weights]
        elif ctype == 5123:
            weights = [[w/65535.0 for w in wv] for wv in weights]
        
        if 'indices' in prim:
            indices, _, _ = get_accessor(gltf, bin_data, prim['indices'])
        else:
            indices = list(range(len(pos)))
        
        local_to_new = []
        for vi in range(len(pos)):
            j = joints[vi]; w = weights[vi]
            
            # Remap each joint via glb_skin_to_cs2
            new_j = [0, 0, 0, 0]
            new_w = [0.0, 0.0, 0.0, 0.0]
            valid = True
            kept_weight_total = 0.0
            for k in range(4):
                gi = int(j[k])
                wt = w[k]
                if wt < 0.001:
                    continue
                cs2 = glb_skin_to_cs2.get(gi)
                if cs2 is None:
                    valid = False
                    break
                new_j[k] = cs2
                new_w[k] = wt
                kept_weight_total += wt
            
            if not valid:
                local_to_new.append(-1); dropped_v += 1; continue
            
            # Renormalize weights
            if kept_weight_total > 0.001 and abs(kept_weight_total - 1.0) > 0.001:
                inv = 1.0 / kept_weight_total
                new_w = [w * inv for w in new_w]
            
            # Dedup check
            key = vert_key(pos[vi], norm[vi], new_j, new_w)
            existing = dedup_map.get(key)
            if existing is not None:
                local_to_new.append(existing)
                dedup_hits += 1
                continue
            
            new_idx = len(new_pos)
            dedup_map[key] = new_idx
            local_to_new.append(new_idx)
            new_pos.append(pos[vi])
            new_norm.append(norm[vi])
            new_joints.append(new_j)
            new_weights.append(new_w)
        
        for ti in range(0, len(indices), 3):
            a,b,c = indices[ti], indices[ti+1], indices[ti+2]
            na,nb,nc = local_to_new[a], local_to_new[b], local_to_new[c]
            if na < 0 or nb < 0 or nc < 0:
                dropped_t += 1; continue
            new_indices.extend([na, nb, nc])
    
    if dedup_hits > 0:
        print(f"  Deduplicated {dedup_hits} vertices")
    
    vc = len(new_pos)
    ic = len(new_indices)
    bc = max_bones
    
    if dropped_v > 0:
        print(f"  Dropped {dropped_v} verts, {dropped_t} tris (unmappable joints)")
    print(f"  Final: {vc} verts, {ic} indices, {bc} bones")
    if vc > 65535:
        print(f"  WARNING: vc > 65535 — chams .bin uses u16 indices!")
    
    # === WRITE OUTPUT ===
    out = bytearray()
    out += struct.pack('<III', vc, ic, bc)
    
    for i in range(vc):
        p = new_pos[i]; n = new_norm[i]; j = new_joints[i]; w = new_weights[i]
        out += struct.pack('<fff', p[0], p[1], p[2])
        out += struct.pack('<fff', n[0], n[1], n[2])
        out += struct.pack('<ffff', float(j[0]), float(j[1]), float(j[2]), float(j[3]))
        out += struct.pack('<ffff', w[0], w[1], w[2], w[3])
    
    for idx in new_indices:
        out += struct.pack('<H', idx)
    if ic % 2 != 0:
        out += b'\x00\x00'
    
    # === IBMs ===
    # IBM[0] = M_root rotation, normalized, no translation (the synthetic root anchor)
    rot = [[M_root[i][j] for j in range(3)] for i in range(3)]
    for i in range(3):
        L = (rot[i][0]**2 + rot[i][1]**2 + rot[i][2]**2) ** 0.5
        if L > 1e-9:
            for j in range(3):
                rot[i][j] /= L
    for row in range(3):
        for col in range(3):
            out += struct.pack('<f', rot[row][col])
        out += struct.pack('<f', 0.0)
    
    # IBM[1..bc-1] = inverse(joint_world_in_cs2_space) for each CS2 bone slot
    for cs2_idx in range(1, bc):
        glb_node = cs2_to_glb_node[cs2_idx]
        if glb_node is None:
            # Identity for unmapped bones
            for r in range(3):
                for c in range(4):
                    out += struct.pack('<f', 1.0 if r == c else 0.0)
            continue
        JW_glb = joint_world(gltf, parent_map, glb_node)
        JW_cs2 = mat_mul(M_root_inv, JW_glb)
        IBM = mat_inv(JW_cs2)
        for row in range(3):
            for col in range(4):
                out += struct.pack('<f', IBM[row][col])
    
    with open(bin_path, 'wb') as f:
        f.write(bytes(out))
    print(f"Wrote {bin_path}: {len(out)} bytes\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--max-bones', type=int, default=94)
    args = ap.parse_args()
    convert(args.input, args.output, args.max_bones)
