"""Signed-distance modelling for organic parts (skulls, a bird's body).

Shapes are numpy SDFs over points P[..., 3]. mesh() samples one on a grid, extracts the zero level set
with marching cubes, decimates it to a triangle budget and smart-UV-projects it, returning an rbxgeo
primitive (verts, faces, per-corner UVs in studs) for Asm.add.
"""
import math

import bpy  # must precede bmesh/mathutils
import bmesh
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes

from rbxgeo import select_only


def open_edges(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    n = sum(1 for e in bm.edges if len(e.link_faces) != 2)
    bm.free()
    return n


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def rotated(P, pivot, R):
    """Points expressed in the frame of a part rotated by R about pivot (evaluate the part there)."""
    pivot = np.asarray(pivot, float)
    return (P - pivot) @ R + pivot


def _loc(P, c, rot=None):
    q = P - np.asarray(c, float)
    return q if rot is None else q @ np.asarray(rot, float)   # rot columns are the local axes


def sphere(P, c, r):
    return np.linalg.norm(_loc(P, c), axis=-1) - r


def ellipsoid(P, c, rad, rot=None):
    """Approximate distance to an ellipsoid with semi-axes rad (exact on the axes)."""
    q = _loc(P, c, rot)
    rad = np.asarray(rad, float)
    k0 = np.linalg.norm(q / rad, axis=-1)
    k1 = np.linalg.norm(q / (rad * rad), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)


def capsule(P, a, b, ra, rb=None):
    """Segment a-b, radius running linearly from ra to rb (a rounded cone when they differ)."""
    a = np.asarray(a, float)
    ba = np.asarray(b, float) - a
    pa = P - a
    h = np.clip((pa @ ba) / float(ba @ ba), 0.0, 1.0)
    r = ra if rb is None else ra + (rb - ra) * h
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - r


def box(P, c, half, r=0.0, rot=None):
    q = np.abs(_loc(P, c, rot)) - (np.asarray(half, float) - r)
    return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(-1), 0.0) - r


def smin(a, b, k=0.0):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k=0.0):
    return -smin(-a, -b, k)


def cut(a, b, k=0.0):
    """a with b carved out."""
    return smax(a, -b, k)


def waves(P, freq, seed=0, octaves=5):
    """Smooth pseudo-noise in about [-1, 1]: randomly oriented sine waves."""
    rng = np.random.default_rng(seed)
    out, amp = np.zeros(P.shape[:-1]), 0.0
    for i in range(octaves):
        d = rng.normal(size=3)
        out += np.sin(P @ (d / np.linalg.norm(d)) * freq * 1.9 ** i + rng.uniform(0, 2 * math.pi)) / 1.7 ** i
        amp += 1 / 1.7 ** i
    return out / amp


def mesh(fn, lo, hi, step, tris, sym=None, angle=60.0, name='sdf'):
    """Closed mesh of fn's zero level set inside lo..hi: marching cubes, collapse decimation to about
    `tris` triangles (mirror-symmetric about local X=0 when sym='X'), smart UV project.
    Returns (verts, faces, uvs) with UVs in model units."""
    lo, hi = np.asarray(lo, float) - step, np.asarray(hi, float) + step
    axes = [np.arange(lo[i], hi[i] + step * 0.5, step) for i in range(3)]
    d = fn(np.stack(np.meshgrid(*axes, indexing='ij'), -1))
    inner = d[1:-1, 1:-1, 1:-1]
    if min(inner[0].min(), inner[-1].min(), inner[:, 0].min(), inner[:, -1].min(),
           inner[:, :, 0].min(), inner[:, :, -1].min()) < 0:
        print(f'WARN sdf {name}: surface clipped by the sampling box')
    d[[0, -1]] = step
    d[:, [0, -1]] = step
    d[:, :, [0, -1]] = step
    v, f, _, _ = marching_cubes(d, 0.0, spacing=(step, step, step))
    # Tiny detached specks (noise islands, cut-off tooth tips) would collapse into degenerate shells.
    ncomp, lab = connected_components(coo_matrix((np.ones(3 * len(f)), (f.ravel(), np.roll(f, 1, 1).ravel())),
                                                 shape=(len(v), len(v))), directed=False)
    if ncomp > 1:
        size = np.bincount(lab[f[:, 0]], minlength=ncomp)
        fkeep = (size >= max(64, 0.01 * len(f)))[lab[f[:, 0]]]
        used = np.unique(f[fkeep])
        remap = np.full(len(v), -1)
        remap[used] = np.arange(len(used))
        print(f'SDF {name}: dropped {ncomp - int((size >= max(64, 0.01 * len(f))).sum())} specks')
        v, f = v[used], remap[f[fkeep]]
    me = bpy.data.meshes.new(name)
    me.from_pydata((v + lo).tolist(), [], f.tolist())
    me.validate()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    raw = len(me.polygons)
    if tris < raw:
        # Collapse can fold thin features (teeth, gaps) into duplicate or non-manifold faces, most often on
        # the mirror plane; nudge the ratio, then drop the symmetry, until the result is clean and closed.
        for attempt in range(8):
            mod = ob.modifiers.new('dec', 'DECIMATE')
            mod.ratio = tris / raw * (1 - 0.02 * (attempt % 4))
            mod.use_collapse_triangulate = True
            if sym and attempt < 4:
                mod.use_symmetry, mod.symmetry_axis = True, sym
            low = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
            ob.modifiers.remove(mod)
            n0 = len(low.polygons)
            low.validate()
            bad = open_edges(low)
            if len(low.polygons) == n0 and bad == 0:
                break
            print(f'SDF {name}: attempt {attempt} dropped {n0 - len(low.polygons)} faces, {bad} open edges; retrying')
            bpy.data.meshes.remove(low)
        else:
            raise RuntimeError(f'SDF {name}: no clean decimation found')
        ob.data = low
        bpy.data.meshes.remove(me)
        me = low
    select_only(ob)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(angle), island_margin=0.0, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    uv = np.empty(len(me.loops) * 2)
    me.uv_layers.active.data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2)
    faces, uvs, a3, a2 = [], [], 0.0, 0.0
    for p in me.polygons:
        q = uv[p.loop_start:p.loop_start + p.loop_total]
        faces.append(list(p.vertices))
        uvs.append(q)
        a3 += p.area
        a2 += 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
    s = math.sqrt(a3 / max(a2, 1e-12))   # back to model units, so packing keeps texel density uniform
    print(f'SDF {name}: {raw} -> {len(faces)} faces')
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    return co.reshape(-1, 3), faces, [[(float(a) * s, float(b) * s) for a, b in q] for q in uvs]
