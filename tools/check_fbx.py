"""Re-import exported accessory FBX files in a clean scene and check them against the Roblox limits and
their build report (<name>.json beside the FBX). Exits non-zero when any check fails.

    /opt/rbxenv/bin/python tools/check_fbx.py assets/<Name>/<Name>.fbx [...]
"""
import json
import os
import sys

import bpy  # noqa: F401  (must precede mathutils)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbxkit as K  # noqa: E402


def check(path):
    K.new_scene()
    bpy.ops.import_scene.fbx(filepath=os.path.abspath(path))
    obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    with open(os.path.splitext(path)[0] + '.json') as fh:
        rep = json.load(fh)
    problems = []
    if len(obs) != 1:
        problems.append(f'{len(obs)} mesh objects (expected 1)')
    ob = obs[0]
    bpy.context.view_layer.update()
    st = K.mesh_stats(ob)
    size, centre = st['max'] - st['min'], (st['max'] + st['min']) / 2
    tex = {}
    for m in ob.data.materials:
        for n in m.node_tree.nodes if m and m.node_tree else []:
            if n.type == 'TEX_IMAGE' and n.image is not None:
                tex[os.path.basename(n.image.filepath or n.image.name)] = list(n.image.size)
    alb = [k for k in tex if '_ALB' in k]
    pbr = [k for k in tex if any(s in k for s in ('_NOR', '_RGH', '_MTL'))]
    if st['tris'] > K.TRI_LIMIT:
        problems.append(f'{st["tris"]} triangles > {K.TRI_LIMIT}')
    if st['open_edges'] or st['degenerate_faces']:
        problems.append(f'not watertight: {st["open_edges"]} open edges, {st["degenerate_faces"]} degenerate faces')
    if st['materials'] != 1 or st['uv_layers'] < 1:
        problems.append(f'{st["materials"]} materials / {st["uv_layers"]} UV layers (expected 1 / >=1)')
    if np.abs(size - np.array(rep['size_studs'])).max() > 0.005:
        problems.append(f'size {np.round(size, 3).tolist()} differs from report {rep["size_studs"]}')
    if np.abs(centre).max() > 0.005:
        problems.append(f'mesh not centred on its origin: {np.round(centre, 4).tolist()}')
    if len(alb) != 1 or len(pbr) != 3:
        problems.append(f'expected 1 colour + 3 PBR textures embedded, found {sorted(tex)}')
    if any(max(tex[k]) > K.COLOR_SIZE for k in alb) or any(max(tex[k]) > K.PBR_SIZE for k in pbr):
        problems.append(f'texture over size limit: {tex}')
    for key in ('tri_ok', 'watertight', 'bounds_ok'):
        if not rep.get(key):
            problems.append(f'report {key} is false')
    if rep.get('body_inside'):
        problems.append(f'report body_inside: {rep["body_inside"]}')
    out = {'file': path, 'ok': not problems, 'problems': problems, 'tris': st['tris'],
           'size_studs': np.round(size, 3).tolist(), 'textures': tex, 'type': rep.get('type'),
           'attachment': rep.get('roblox_attachment', rep.get('attachment')), 'rig': rep.get('rig'),
           'min_clearance': rep.get('min_clearance')}
    print('FBXCHECK', json.dumps(out))
    return not problems


if __name__ == '__main__':
    ok = [check(p) for p in sys.argv[1:]]
    sys.exit(0 if ok and all(ok) else 1)
