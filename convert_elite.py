#!/usr/bin/env python3
"""
Convert dual berettas GLB into a single-beretta .bin, properly baked into
the correct bone's local space using actual bone weights from the GLB.

The GLB has this bone hierarchy:
  weapon (root)
  ├── weapon_hand_l (joint 1) → weapon_l (joint 2) = left gun
  └── weapon_hand_r (joint 4) → weapon_r (joint 5) = right gun

This script:
1. Reads bone weights per vertex
2. Splits vertices into left-gun vs right-gun based on primary bone
3. Exports the right gun baked into weapon_r's local space (using IBM5)
4. At runtime, the script reads bone 5 from the weapon entity's bone array

Output: weapon_pist_elite_single.bin (right gun only, in weapon_r local space)
"""
import struct
import json
import math
import sys
import os

# Import shared functions from the main converter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from glb_to_weapon_bin import (parse_glb, get_accessor, colmajor_to_rowmajor,
                                 transform_point, transform_normal, normalize,
                                 get_glb_ibms)


def convert_elite(glb_path, out_path):
    gltf, bin_data = parse_glb(glb_path)

    # Find the HD mesh (preferred) or legacy
    mesh_idx = None
    skin_idx = None
    for i, m in enumerate(gltf.get('meshes', [])):
        name = m.get('name', '').lower()
        if '_hd' in name or 'body_hd' in name:
            for n in gltf.get('nodes', []):
                if n.get('mesh') == i and 'skin' in n:
                    mesh_idx = i
                    skin_idx = n['skin']
                    print(f"  Using HD mesh: {m.get('name','?')}")
                    break
        if mesh_idx is not None:
            break
    if mesh_idx is None:
        for i, m in enumerate(gltf.get('meshes', [])):
            for n in gltf.get('nodes', []):
                if n.get('mesh') == i and 'skin' in n:
                    mesh_idx = i
                    skin_idx = n['skin']
                    print(f"  Using mesh: {m.get('name','?')}")
                    break
            if mesh_idx is not None:
                break

    mesh = gltf['meshes'][mesh_idx]
    skin = gltf['skins'][skin_idx]

    joint_names = [gltf['nodes'][j].get('name', f'joint_{ji}')
                   for ji, j in enumerate(skin['joints'])]
    print(f"  Joints: {joint_names}")

    # Get IBMs
    ibms = get_glb_ibms(gltf, bin_data, skin)
    if ibms is None:
        print("ERROR: No inverse bind matrices"); return

    # Identify right-gun joints: weapon_hand_r (4), weapon_r (5), slide_r (6),
    # hammer_r (10), magazine_r (11), trigger_r (12), ag1_hand_r (15)
    right_joints = set()
    left_joints = set()
    for ji, name in enumerate(joint_names):
        lower = name.lower()
        if '_r' in lower and ('weapon' in lower or 'slide' in lower or 'hammer' in lower
                              or 'magazine' in lower or 'trigger' in lower or 'ag1' in lower):
            right_joints.add(ji)
        elif '_l' in lower and ('weapon' in lower or 'slide' in lower or 'hammer' in lower
                                or 'magazine' in lower or 'trigger' in lower or 'ag1' in lower):
            left_joints.add(ji)

    print(f"  Right-gun joints: {right_joints} = {[joint_names[j] for j in sorted(right_joints)]}")
    print(f"  Left-gun joints: {left_joints} = {[joint_names[j] for j in sorted(left_joints)]}")

    # Find the weapon_r joint index for the IBM
    weapon_r_idx = None
    for ji, name in enumerate(joint_names):
        if name.lower() == 'weapon_r':
            weapon_r_idx = ji
            break
    if weapon_r_idx is None:
        print("ERROR: Could not find weapon_r joint"); return

    weapon_l_idx = None
    for ji, name in enumerate(joint_names):
        if name.lower() == 'weapon_l':
            weapon_l_idx = ji
            break

    print(f"  weapon_r = joint {weapon_r_idx}, weapon_l = joint {weapon_l_idx}")

    ibm_r = ibms[weapon_r_idx]
    ibm_l = ibms[weapon_l_idx] if weapon_l_idx is not None else None

    # Process vertices — split by bone weights
    new_pos_r, new_norm_r, new_indices_r = [], [], []
    new_pos_l, new_norm_l, new_indices_l = [], [], []
    dedup_r, dedup_l = {}, {}

    def vert_key(p, n):
        return (round(p[0]*10000), round(p[1]*10000), round(p[2]*10000),
                round(n[0]*1000), round(n[1]*1000), round(n[2]*1000))

    for prim in mesh['primitives']:
        attrs = prim['attributes']
        pos, _, _ = get_accessor(gltf, bin_data, attrs['POSITION'])
        norm, _, _ = get_accessor(gltf, bin_data, attrs.get('NORMAL', attrs['POSITION']))

        # Read bone weights and indices
        joints_data = None
        weights_data = None
        if 'JOINTS_0' in attrs:
            joints_data, _, _ = get_accessor(gltf, bin_data, attrs['JOINTS_0'])
        if 'WEIGHTS_0' in attrs:
            weights_data, _, _ = get_accessor(gltf, bin_data, attrs['WEIGHTS_0'])

        if 'indices' in prim:
            indices, _, _ = get_accessor(gltf, bin_data, prim['indices'])
        else:
            indices = list(range(len(pos)))

        # Classify each vertex as left or right based on primary bone
        vert_side = []  # 'r' or 'l' per vertex
        local_to_new_r = {}
        local_to_new_l = {}

        for vi in range(len(pos)):
            # Determine primary bone
            side = 'r'  # default to right
            if joints_data and weights_data:
                j = joints_data[vi]
                w = weights_data[vi]
                # Find highest-weight bone
                max_w = 0
                primary_joint = 0
                for bi in range(4):
                    if w[bi] > max_w:
                        max_w = w[bi]
                        primary_joint = j[bi]
                if primary_joint in left_joints:
                    side = 'l'
                elif primary_joint in right_joints:
                    side = 'r'
                else:
                    # Root bone (weapon) — assign based on position Y
                    # Left gun is at positive Y in mesh space
                    side = 'l' if pos[vi][1] > 0 else 'r'

            vert_side.append(side)

            # Transform into the appropriate bone's local space
            if side == 'r':
                tp = transform_point(ibm_r, pos[vi])
                tn = normalize(transform_normal(ibm_r, norm[vi]))
                key = vert_key(tp, tn)
                if key not in dedup_r:
                    dedup_r[key] = len(new_pos_r)
                    new_pos_r.append(tp)
                    new_norm_r.append(tn)
                local_to_new_r[vi] = dedup_r[key]
            else:
                if ibm_l:
                    tp = transform_point(ibm_l, pos[vi])
                    tn = normalize(transform_normal(ibm_l, norm[vi]))
                else:
                    tp = list(pos[vi])
                    tn = normalize(list(norm[vi]))
                key = vert_key(tp, tn)
                if key not in dedup_l:
                    dedup_l[key] = len(new_pos_l)
                    new_pos_l.append(tp)
                    new_norm_l.append(tn)
                local_to_new_l[vi] = dedup_l[key]

        # Build triangles — only keep triangles where ALL verts are same side
        for ti in range(0, len(indices), 3):
            a, b, c = indices[ti], indices[ti+1], indices[ti+2]
            sa, sb, sc = vert_side[a], vert_side[b], vert_side[c]
            if sa == sb == sc == 'r':
                new_indices_r.extend([local_to_new_r[a], local_to_new_r[b], local_to_new_r[c]])
            elif sa == sb == sc == 'l':
                new_indices_l.extend([local_to_new_l[a], local_to_new_l[b], local_to_new_l[c]])
            # Mixed triangles at the boundary — duplicate to both sides
            # (rare, happens at shared edges)

    print(f"  Right gun: {len(new_pos_r)} verts, {len(new_indices_r)} indices")
    print(f"  Left gun: {len(new_pos_l)} verts, {len(new_indices_l)} indices")

    def write_bin(path, positions, normals, indices_list):
        vc = len(positions)
        ic = len(indices_list)
        use_32bit = vc > 65535
        bc_flags = 1 | (0x80000000 if use_32bit else 0)

        out = bytearray()
        out += struct.pack('<III', vc, ic, bc_flags)

        for i in range(vc):
            p = positions[i]; n = normals[i]
            out += struct.pack('<fff', p[0], p[1], p[2])
            out += struct.pack('<fff', n[0], n[1], n[2])
            out += struct.pack('<ffff', 0.0, 0.0, 0.0, 0.0)  # bone indices
            out += struct.pack('<ffff', 1.0, 0.0, 0.0, 0.0)  # weights

        if use_32bit:
            for idx in indices_list:
                out += struct.pack('<I', idx)
        else:
            for idx in indices_list:
                out += struct.pack('<H', idx)
            if ic % 2 != 0:
                out += b'\x00\x00'

        # IBM: identity (vertices already in bone local space)
        for r in range(3):
            for c in range(4):
                out += struct.pack('<f', 1.0 if r == c else 0.0)

        with open(path, 'wb') as f:
            f.write(bytes(out))
        print(f"  Wrote {path}: {len(out)} bytes")

    # Write right gun
    base = os.path.splitext(out_path)[0]
    write_bin(out_path, new_pos_r, new_norm_r, new_indices_r)

    # Also write left gun
    left_path = base + "_left.bin"
    write_bin(left_path, new_pos_l, new_norm_l, new_indices_l)

    # Print bone indices for runtime reference
    print(f"\n  === RUNTIME INFO ===")
    print(f"  Right gun bone index: {weapon_r_idx} (offset {weapon_r_idx * 32} in bone array)")
    print(f"  Left gun bone index: {weapon_l_idx} (offset {weapon_l_idx * 32} in bone array)")


if __name__ == '__main__':
    glb = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\user1\Desktop\weaponmodels\weapon_pist_elite.glb"
    out = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\user1\Documents\My Games\chams_models\weapon_pist_elite_single.bin"
    convert_elite(glb, out)
