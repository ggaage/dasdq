"""Shared pipeline for procedurally built Roblox rigid accessories (Blender 5.2 `bpy` module).

Scale and axes follow Roblox's own reference FBX files: 1 Blender unit = 1 stud, scene unit
scale 0.01 so the exported FBX has UnitScaleFactor 1, the character faces -Y and Z is up.
Geometry primitives live in rbxgeo; this module adds materials, baking, validation and export.
"""
import json
import os
import sys
import urllib.request

import bpy  # must precede bmesh/mathutils: the bpy module sets up their import paths
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

import rbxgeo
from rbxgeo import (CHUNK, Path, _unit, axes, basis, box, chain_frames, cross_outline, ellipse,  # noqa: F401
                    fillet, frame, frames, inset, lathe, mat_trs, pack_uv, point_in_poly, prism, ring, sag,
                    select_only, solid_from_surface, sphere, spline, sweep_profile, triangulate_region, tube,
                    uv_coverage, xf)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, 'assets')
CACHE = os.path.join(REPO, '.cache')
MANNEQUIN_URL = ('https://raw.githubusercontent.com/Roblox/avatar/main/Legacy/LayeredClothingDeveloperRelease/'
                 'Layered_Clothing_Examples/LC_ClassicMale/LC_ClassicMale.fbx')

# Attachment points of Roblox's LC_ClassicMale reference mannequin, in studs (bone + FBX offsets).
ATTACH_LC = {
    'Hat': (0.0, 0.0, 2.913), 'FaceFront': (0.0, -0.598, 2.262), 'FaceCenter': (0.0, 0.0, 2.371),
    'Neck': (0.0, 0.0, 1.723), 'WaistCenter': (0.0, 0.0, 0.0), 'WaistFront': (0.0, -0.627, 0.199),
    'WaistBack': (0.0, 0.583, 0.289), 'LeftCollar': (0.716, 0.0, 1.722), 'RightCollar': (-0.58, 0.0, 1.722),
    'BodyBack': (0.0, 0.599, 0.983), 'Hair': (0.0, 0.0, 2.913), 'RightShoulder': (-0.984, -0.169, 1.707),
}
# Roblox's default R6 rig with WaistCenter at the origin: Torso 2x2x1 centred at z=1, limbs 1x2x1, the
# 1.2-stud classic head centred 1.5 above the torso centre. Offsets are the R6 rig's own attachments
# converted to these axes (Roblox x -> -x, Roblox z -> y); +X is the character's left.
ATTACH_R6 = {
    'Hat': (0.0, 0.0, 3.1), 'Hair': (0.0, 0.0, 3.1), 'FaceFront': (0.0, -0.6, 2.5), 'FaceCenter': (0.0, 0.0, 2.5),
    'Neck': (0.0, 0.0, 2.0), 'BodyFront': (0.0, -0.5, 1.0), 'BodyBack': (0.0, 0.5, 1.0),
    'LeftCollar': (1.0, 0.0, 2.0), 'RightCollar': (-1.0, 0.0, 2.0), 'WaistFront': (0.0, -0.5, 0.0),
    'WaistCenter': (0.0, 0.0, 0.0), 'WaistBack': (0.0, 0.5, 0.0), 'LeftShoulder': (1.5, 0.0, 2.0),
    'RightShoulder': (-1.5, 0.0, 2.0),
}
R6_PARTS = {'Torso': ((0, 0, 1), (2, 1, 2)), 'LeftArm': ((1.5, 0, 1), (1, 1, 2)), 'RightArm': ((-1.5, 0, 1), (1, 1, 2)),
            'LeftLeg': ((0.5, 0, -1), (1, 1, 2)), 'RightLeg': ((-0.5, 0, -1), (1, 1, 2))}
RIG = 'LC'
ATTACH = dict(ATTACH_LC)


def use_rig(name):
    """Reference body for fitting, checks and previews: 'LC' (LC_ClassicMale, default) or 'R6'."""
    global RIG
    RIG = name
    ATTACH.clear()
    ATTACH.update(ATTACH_R6 if name == 'R6' else ATTACH_LC)


# Strictest of the Classic / Normal / Slender rigid accessory limits (Creator Hub) relative to the
# attachment point, as (x half-width, y to the front, y to the back, z below, z above).
LIMITS = {
    'Hat': (0.89, 0.89, 0.89, 1.25, 1.25),
    'Face': (0.89, 0.59, 0.59, 0.625, 0.625),
    'Waist': (1.88, 3.365, 3.365, 1.885, 1.414),
    'Neck': (1.295, 0.96, 0.96, 1.5, 1.5),
    'Shoulder': (1.295, 1.44, 1.44, 1.5, 1.5),      # on Left/RightCollarAttachment
    'ShoulderArm': (1.185, 1.375, 1.375, 1.5, 1.5),  # on Left/RightShoulderAttachment (rides the upper arm)
    'Hair': (0.89, 0.892, 1.189, 1.875, 1.25),
    'Back': (4.32, 1.443, 2.886, 3.5, 3.5),
}
TRI_LIMIT = 4000
COLOR_SIZE, PBR_SIZE = 1024, 256   # Marketplace PBR limits: ColorMap 1024, other maps 256


# --------------------------------------------------------------------------------------- scene
def new_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    us = sc.unit_settings
    us.system, us.scale_length, us.length_unit = 'METRIC', 0.01, 'CENTIMETERS'
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.world = bpy.data.worlds.new('World')
    return sc


# ----------------------------------------------------------------------------------- mannequin
def ensure_mannequin():
    """Cached reference body for the current rig (studs). Resets the current file when it builds."""
    return _ensure_r6() if RIG == 'R6' else _ensure_lc()


def _ensure_r6():
    """R6 body: sharp 1x2x1 / 2x2x1 blocks plus LC_ClassicMale's classic head moved to the R6 head position."""
    path = os.path.join(CACHE, 'mannequin_r6.blend')
    if os.path.exists(path):
        return path
    lc = _ensure_lc()
    new_scene()
    with bpy.data.libraries.load(lc) as (src, dst):
        dst.objects = ['MQ_Head']
    head = dst.objects[0]
    me = head.data.copy()
    me.transform(head.matrix_world)
    v = np.array([x.co[:] for x in me.vertices])
    centre = (v.min(0) + v.max(0)) / 2
    me.transform(Matrix.Translation(Vector(ATTACH_R6['FaceCenter']) - Vector(centre)))
    mat = head.data.materials[0]
    bpy.data.objects.remove(head)
    keep = [bpy.data.objects.new('MQ_Head', me)]
    for name, (c, s) in R6_PARTS.items():
        bv, bf, _ = box(*s)
        bme = bpy.data.meshes.new(name)
        bme.from_pydata((np.asarray(bv) + c).tolist(), [], bf)
        bme.materials.append(mat)
        keep.append(bpy.data.objects.new('MQ_' + name, bme))
    for o in keep:
        bpy.context.scene.collection.objects.link(o)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    return path


def _ensure_lc():
    """Static copy of Roblox's LC_ClassicMale (studs). Resets the current file when it builds."""
    path = os.path.join(CACHE, 'mannequin.blend')
    if os.path.exists(path):
        return path
    os.makedirs(CACHE, exist_ok=True)
    fbx = os.path.join(CACHE, 'LC_ClassicMale.fbx')
    if not os.path.exists(fbx):
        urllib.request.urlretrieve(MANNEQUIN_URL, fbx)
    new_scene()
    bpy.ops.import_scene.fbx(filepath=fbx)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    keep = []
    for o in list(bpy.data.objects):
        if o.type == 'MESH' and o.name.endswith('_Geo'):
            oe = o.evaluated_get(dg)
            me = bpy.data.meshes.new_from_object(oe)
            me.transform(oe.matrix_world)   # bake the armature/world transform into the vertices
            keep.append(bpy.data.objects.new('MQ_' + o.name[:-4], me))
    names = {k.name for k in keep}
    for o in [o for o in bpy.data.objects if o.name not in names]:
        bpy.data.objects.remove(o)
    mat = bpy.data.materials.new('Mannequin')
    b = mat.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (0.018, 0.018, 0.02, 1)
    b.inputs['Roughness'].default_value = 0.7
    for n in keep:
        bpy.context.scene.collection.objects.link(n)
        n.data.materials.clear()
        n.data.materials.append(mat)
        for p in n.data.polygons:
            p.use_smooth = True
    bpy.ops.wm.save_as_mainfile(filepath=path)
    return path


def load_mannequin(parts=None):
    path = ensure_mannequin()
    with bpy.data.libraries.load(path) as (src, dst):
        dst.objects = [n for n in src.objects if parts is None or n[3:] in parts]
    for o in dst.objects:
        bpy.context.scene.collection.objects.link(o)
    return dst.objects


def mannequin_verts(part):
    ob = load_mannequin([part])[0]
    m = np.array(ob.matrix_world)
    v = np.array([vv.co[:] for vv in ob.data.vertices])
    v = (np.c_[v, np.ones(len(v))] @ m.T)[:, :3]
    bpy.data.objects.remove(ob)
    return v


def polar_profile(verts, z0, z1, center=(0.0, 0.0), bins=72, smooth=2):
    """Max horizontal radius of `verts` within [z0, z1] per angle. Angle 0 = front (-Y), +pi/2 = +X."""
    v = verts[(verts[:, 2] >= z0) & (verts[:, 2] <= z1)]
    x, y = v[:, 0] - center[0], v[:, 1] - center[1]
    th = np.mod(np.arctan2(x, -y), 2 * np.pi)
    r = np.hypot(x, y)
    idx = (th / (2 * np.pi) * bins).astype(int) % bins
    prof = np.full(bins, np.nan)
    for i in range(bins):
        sel = r[idx == i]
        if len(sel):
            prof[i] = sel.max()
    ok = ~np.isnan(prof)
    xs = np.arange(bins)
    prof = np.interp(xs, xs[ok], prof[ok], period=bins)
    if smooth:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        prof = np.convolve(np.r_[prof[-smooth:], prof, prof[:smooth]], k, 'valid')

    def f(theta):
        t = np.mod(np.asarray(theta), 2 * np.pi) / (2 * np.pi) * bins
        return np.interp(t, np.arange(bins + 1), np.r_[prof, prof[0]])
    return f


class Asm(rbxgeo.Asm):
    def __init__(self):
        super().__init__()
        self.groups, self.part_group = {}, []

    def part(self, group, prim, mat, m=None, smooth=180.0, uvs=1.0, vfac=None):
        """add() that also tallies triangles per named group."""
        self.groups[group] = self.groups.get(group, 0) + sum(len(f) - 2 for f in prim[1])
        self.part_group.append(group)
        return self.add(prim, mat, m, smooth=smooth, uvs=uvs, vfac=vfac)

    def build(self, name):
        return super().build(name, material)


def safety_pin(asm, group, top, down, normal, length=0.12, wire=0.0032, mat='silver'):
    """Closed safety pin in the plane normal to `normal`: spring coil at `top`, clasp `length` along `down`."""
    down = _unit(down)
    normal = _unit(np.asarray(normal, float) - down * np.dot(normal, down))
    side = np.cross(normal, down)
    rc = 0.0085
    o = np.asarray(top, float) + down * rc
    for k in (-1, 1):
        asm.part(group, ring(rc, rc, wire, segs=8, sides=3), mat, frame(o + normal * k * wire * 0.9, down, normal))
    asm.part(group, tube([o - side * rc, o - side * rc + down * length], wire, sides=4), mat)
    asm.part(group, tube([o + side * rc, o + side * 0.003 + down * (length - 0.006)], wire, sides=4), mat)
    asm.part(group, box(0.026, 0.024, 0.012), mat, axes(o + down * (length + 0.004), side, -down), smooth=40)


def push_out(prim, trees, gap=0.004):
    """Move vertices of a world-space primitive that sit inside, or within `gap` of, any BVH surface out
    to `gap` along that surface's normal."""
    v = np.array(prim[0], float)
    for i, p in enumerate(v):
        for t in trees:
            loc, nrm, _, _ = t.find_nearest(Vector(p))
            if loc is not None and (Vector(p) - loc).dot(nrm) < gap:
                v[i] = np.array(loc + nrm * gap)
    return (v, *prim[1:])


def placed(prim, m):
    """World vertices of a primitive under the 4x4 m."""
    v = np.asarray(prim[0], float)
    return (np.c_[v, np.ones(len(v))] @ np.asarray(m).T)[:, :3]


def body_check(verts, labels=None):
    """Vertices inside the mannequin at rest pose (per body part, and per label when given), and the
    smallest clearance of the others."""
    parts = load_mannequin()
    dg = bpy.context.evaluated_depsgraph_get()
    hits, gap = {}, float('inf')
    for o in parts:
        tree = BVHTree.FromObject(o, dg)
        for i, v in enumerate(verts):
            loc, nrm, _, dist = tree.find_nearest(Vector(v))
            if loc is None:
                continue
            if (Vector(v) - loc).dot(nrm) < 0:
                key = o.name if labels is None else f'{o.name}:{labels[i]}'
                hits[key] = hits.get(key, 0) + 1
            else:
                gap = min(gap, dist)
    for o in parts:
        bpy.data.objects.remove(o)
    return hits, gap


# ----------------------------------------------------------------------------------- materials
def _sock(socks, ident):
    for s in socks:
        if s.identifier == ident:
            return s
    return socks[ident]


class NB:
    """Tiny node-graph builder for procedural (bake-only) materials."""

    def __init__(self, mat):
        self.mat = mat
        self.nt = mat.node_tree
        self.N, self.L = self.nt.nodes, self.nt.links
        for n in list(self.N):
            self.N.remove(n)
        self._tc = None

    def node(self, typ, **props):
        n = self.N.new(typ)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def feed(self, sock, val):
        if isinstance(val, bpy.types.NodeSocket):
            self.L.new(val, sock)
            return
        dv = sock.default_value
        if hasattr(dv, '__len__'):
            if not hasattr(val, '__len__'):
                val = [val] * len(dv)
            elif len(val) == 3 and len(dv) == 4:
                val = (*val, 1.0)
        sock.default_value = val

    def rgb(self, c):
        n = self.node('ShaderNodeRGB')
        n.outputs[0].default_value = (*c, 1.0)
        return n.outputs[0]

    def math(self, op, a, b=0.0, clamp=False):
        n = self.node('ShaderNodeMath', operation=op, use_clamp=clamp)
        self.feed(n.inputs[0], a)
        self.feed(n.inputs[1], b)
        return n.outputs[0]

    def add(self, a, b): return self.math('ADD', a, b)
    def mul(self, a, b): return self.math('MULTIPLY', a, b)
    def sub(self, a, b): return self.math('SUBTRACT', a, b)
    def inv(self, a): return self.math('SUBTRACT', 1.0, a)
    def maxf(self, a, b): return self.math('MAXIMUM', a, b)
    def minf(self, a, b): return self.math('MINIMUM', a, b)
    def absf(self, a): return self.math('ABSOLUTE', a)
    def frac(self, a): return self.math('FRACT', a)

    def xyz(self, vec):
        n = self.node('ShaderNodeSeparateXYZ')
        self.feed(n.inputs[0], vec)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def mapr(self, v, a, b, c=0.0, d=1.0, smooth=False):
        n = self.node('ShaderNodeMapRange', clamp=True, interpolation_type='SMOOTHSTEP' if smooth else 'LINEAR')
        self.feed(n.inputs['Value'], v)
        for k, x in (('From Min', a), ('From Max', b), ('To Min', c), ('To Max', d)):
            n.inputs[k].default_value = x
        return n.outputs['Result']

    def mixc(self, t, a, b, blend='MIX'):
        n = self.node('ShaderNodeMix', data_type='RGBA', blend_type=blend, clamp_result=True)
        self.feed(_sock(n.inputs, 'Factor_Float'), t)
        self.feed(_sock(n.inputs, 'A_Color'), a)
        self.feed(_sock(n.inputs, 'B_Color'), b)
        return _sock(n.outputs, 'Result_Color')

    def coord(self, kind='Object'):
        if self._tc is None:
            self._tc = self.node('ShaderNodeTexCoord')
        return self._tc.outputs[kind]

    def mapping(self, vec, scale=(1, 1, 1), loc=(0, 0, 0), rot=(0, 0, 0)):
        n = self.node('ShaderNodeMapping')
        self.feed(n.inputs['Vector'], vec)
        n.inputs['Location'].default_value = loc
        n.inputs['Rotation'].default_value = rot
        n.inputs['Scale'].default_value = scale
        return n.outputs[0]

    def noise(self, vec, scale, detail=4.0, rough=0.5, distortion=0.0, out='Fac'):
        n = self.node('ShaderNodeTexNoise', noise_dimensions='3D')
        self.feed(n.inputs['Vector'], vec)
        for k, x in (('Scale', scale), ('Detail', detail), ('Roughness', rough), ('Distortion', distortion)):
            n.inputs[k].default_value = x
        return n.outputs[out]

    def voronoi(self, vec, scale, feature='F1', out='Distance', randomness=1.0):
        n = self.node('ShaderNodeTexVoronoi', feature=feature)
        self.feed(n.inputs['Vector'], vec)
        n.inputs['Scale'].default_value = scale
        n.inputs['Randomness'].default_value = randomness
        return n.outputs[out]

    def ao(self, dist, inside=False, samples=8):
        n = self.node('ShaderNodeAmbientOcclusion', inside=inside, samples=samples, only_local=True)
        n.inputs['Distance'].default_value = dist
        return n.outputs['AO']

    def attr(self, name):
        return self.node('ShaderNodeAttribute', attribute_name=name, attribute_type='GEOMETRY').outputs['Fac']

    def image(self, img, vec, extension='CLIP', out='Color'):
        n = self.node('ShaderNodeTexImage', image=img, extension=extension, interpolation='Cubic')
        self.feed(n.inputs['Vector'], vec)
        return n.outputs[out]

    def bump(self, h, strength, dist, normal=None):
        n = self.node('ShaderNodeBump')
        self.feed(n.inputs['Height'], h)
        n.inputs['Strength'].default_value = strength
        n.inputs['Distance'].default_value = dist
        if normal is not None:
            self.feed(n.inputs['Normal'], normal)
        return n.outputs['Normal']

    def finish(self, alb, rough, metal, normal=None):
        if not isinstance(alb, bpy.types.NodeSocket):
            alb = self.rgb(alb)
        bsdf = self.node('ShaderNodeBsdfPrincipled', name='BSDF')
        self.feed(bsdf.inputs['Base Color'], alb)
        self.feed(bsdf.inputs['Roughness'], rough)
        self.feed(bsdf.inputs['Metallic'], metal)
        if normal is not None:
            self.feed(bsdf.inputs['Normal'], normal)
        out = self.node('ShaderNodeOutputMaterial', name='OUT')
        self.L.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        em = self.node('ShaderNodeEmission', name='BAKE_EMIT')
        em.inputs['Strength'].default_value = 1.0
        rm = self.node('ShaderNodeCombineColor', name='BAKE_RM')
        self.feed(rm.inputs[0], rough)
        self.feed(rm.inputs[1], metal)
        self.feed(rm.inputs[2], 0.0)
        al = self.node('ShaderNodeMix', name='BAKE_ALB', data_type='RGBA')
        self.feed(_sock(al.inputs, 'Factor_Float'), 0.0)
        self.feed(_sock(al.inputs, 'A_Color'), alb)
        return self.mat


def edge_mask(nb, radius=0.005, lo=0.03, hi=0.25):
    """1 on geometric edges: deviation of the Bevel normal from the shading normal."""
    bev = nb.node('ShaderNodeBevel', samples=8)
    bev.inputs['Radius'].default_value = radius
    geo = nb.node('ShaderNodeNewGeometry')
    dot = nb.node('ShaderNodeVectorMath', operation='DOT_PRODUCT')
    nb.L.new(bev.outputs['Normal'], dot.inputs[0])
    nb.L.new(geo.outputs['Normal'], dot.inputs[1])
    return nb.mapr(nb.inv(dot.outputs['Value']), lo, hi)


def metal_recipe(nb, base, tarnish, grime, edge, r_lo, r_hi, metal, bump=0.25, sc=1.0, cav=0.07, extra=None):
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    big = nb.noise(co, 5.0 * sc, 5, 0.6)
    mid = nb.noise(co, 24.0 * sc, 4, 0.55)
    fine = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 6.0)), 90.0 * sc, 2, 0.5)
    open_ = nb.mapr(nb.ao(cav), 0.35, 0.95)                        # 0 in crevices, 1 when open
    edge_m = nb.mul(edge_mask(nb), open_)
    tarn = nb.mapr(nb.add(big, nb.mul(rnd, 0.3)), 0.38, 0.75)
    c = nb.mixc(nb.mul(tarn, 0.85), base, tarnish)
    dirt = nb.mapr(nb.add(nb.inv(open_), nb.mul(mid, 0.35)), 0.12, 0.7)
    c = nb.mixc(dirt, c, grime)
    c = nb.mixc(nb.mul(edge_m, 0.75), c, edge)
    rough = nb.mapr(nb.add(nb.mul(tarn, 0.45), nb.mul(dirt, 0.6)), 0.0, 1.0, r_lo, r_hi)
    rough = nb.sub(rough, nb.mul(edge_m, 0.15))
    met = nb.mapr(dirt, 0.2, 1.0, metal, metal * 0.35)
    h = nb.add(nb.mul(mid, 0.55), nb.mul(fine, 0.45))
    if extra:
        c, rough, met, h = extra(nb, c, rough, met, h)
    return nb.finish(c, rough, met, nb.bump(h, bump, 0.004))


def _silver(nb, extra=None):
    return metal_recipe(nb, (0.30, 0.30, 0.31), (0.06, 0.058, 0.055), (0.006, 0.0055, 0.005),
                        (0.62, 0.62, 0.64), 0.25, 0.72, 1.0, extra=extra)


def _pewter(nb, extra=None):
    return metal_recipe(nb, (0.17, 0.17, 0.175), (0.05, 0.048, 0.046), (0.006, 0.0055, 0.005),
                        (0.42, 0.42, 0.44), 0.35, 0.8, 0.85, extra=extra)


def _iron(nb, extra=None):
    return metal_recipe(nb, (0.045, 0.045, 0.048), (0.024, 0.02, 0.017), (0.004, 0.0035, 0.003),
                        (0.22, 0.22, 0.23), 0.42, 0.85, 0.9, bump=0.35, extra=extra)


def _branch(nb):
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    bark = nb.noise(co, 60.0, 6, 0.65, distortion=1.2)
    streak = nb.noise(co, 14.0, 3, 0.5)
    open_ = nb.mapr(nb.ao(0.05), 0.35, 0.95)
    edge_m = nb.mul(edge_mask(nb, 0.004, 0.02, 0.2), open_)
    c = nb.mixc(nb.mapr(bark, 0.35, 0.65), (0.010, 0.007, 0.0045), (0.032, 0.022, 0.013))
    gold = nb.mapr(nb.add(streak, nb.mul(rnd, 0.35)), 0.5, 0.9)
    c = nb.mixc(nb.mul(gold, 0.8), c, (0.085, 0.055, 0.024))
    ridge = nb.mapr(bark, 0.55, 0.75)
    c = nb.mixc(nb.mul(nb.maxf(edge_m, nb.mul(ridge, 0.6)), 0.7), c, (0.16, 0.105, 0.045))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.9), c, (0.003, 0.002, 0.0015))
    rough = nb.mapr(bark, 0.3, 0.7, 0.6, 0.92)
    return nb.finish(c, rough, 0.0, nb.bump(bark, 0.5, 0.004))


def _leaf(nb):
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    n1 = nb.noise(co, 45.0, 5, 0.6, distortion=0.6)
    c = nb.mixc(nb.mapr(nb.add(n1, nb.mul(rnd, 0.4)), 0.35, 0.95), (0.022, 0.014, 0.007), (0.11, 0.07, 0.03))
    c = nb.mixc(nb.mul(nb.inv(nb.mapr(nb.ao(0.04), 0.35, 0.95)), 0.85), c, (0.004, 0.0025, 0.0015))
    return nb.finish(c, nb.mapr(n1, 0.3, 0.7, 0.65, 0.9), 0.0, nb.bump(n1, 0.35, 0.003))


def _gem(nb):
    co = nb.coord('Object')
    n1 = nb.noise(co, 30.0, 3, 0.5)
    c = nb.mixc(nb.mapr(n1, 0.4, 0.7), (0.004, 0.004, 0.005), (0.02, 0.012, 0.014))
    return nb.finish(c, nb.mapr(n1, 0.3, 0.7, 0.06, 0.16), 0.0)


def _wood(nb):
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    grain = nb.noise(co, 30.0, 6, 0.7, distortion=4.0)
    c = nb.mixc(nb.mapr(nb.add(grain, nb.mul(rnd, 0.2)), 0.35, 0.7), (0.012, 0.008, 0.006), (0.05, 0.032, 0.02))
    open_ = nb.mapr(nb.ao(0.03), 0.35, 0.95)
    edge_m = nb.mul(edge_mask(nb, 0.004), open_)
    c = nb.mixc(nb.mul(edge_m, 0.5), c, (0.12, 0.085, 0.055))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.85), c, (0.004, 0.003, 0.002))
    return nb.finish(c, nb.mapr(grain, 0.3, 0.7, 0.55, 0.85), 0.0, nb.bump(grain, 0.4, 0.004))


def _string(nb):
    co = nb.coord('Object')
    n1 = nb.noise(co, 35.0, 4, 0.6)
    c = nb.mixc(nb.mapr(n1, 0.3, 0.75), (0.30, 0.285, 0.25), (0.11, 0.10, 0.085))
    return nb.finish(c, 0.9, 0.0)


def rust(nb, c, rough, met, h):
    """metal_recipe `extra`: rust patches, heavier in crevices."""
    co = nb.coord('Object')
    patch = nb.noise(co, 16.0, 5, 0.6, distortion=0.5)
    grain = nb.noise(co, 80.0, 3, 0.5)
    crev = nb.inv(nb.mapr(nb.ao(0.03), 0.35, 0.95))
    r = nb.mapr(nb.add(nb.mul(patch, 0.9), nb.mul(crev, 0.45)), 0.55, 0.85)
    rc = nb.mixc(nb.mapr(grain, 0.3, 0.7), (0.035, 0.014, 0.006), (0.10, 0.042, 0.014))
    c = nb.mixc(nb.mul(r, 0.85), c, rc)
    return (c, nb.add(rough, nb.mul(r, 0.35)), nb.sub(met, nb.mul(r, 0.75)),
            nb.add(h, nb.mul(nb.mul(r, grain), 0.8)))


def _leather(nb, extra=None):
    """Scuffed black leather; `extra(nb, c, rough, met, h)` can add prints or stitching."""
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    pores = nb.noise(co, 220.0, 3, 0.55)
    crease = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 3.0)), 22.0, 4, 0.6, distortion=1.2)
    open_ = nb.mapr(nb.ao(0.03), 0.35, 0.95)
    wear = nb.mul(edge_mask(nb, 0.004, 0.03, 0.25), open_)
    c = nb.mixc(nb.mapr(nb.add(crease, nb.mul(rnd, 0.3)), 0.3, 0.85), (0.006, 0.006, 0.007), (0.02, 0.019, 0.02))
    c = nb.mixc(nb.mul(wear, 0.75), c, (0.075, 0.072, 0.07))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.8), c, (0.002, 0.002, 0.002))
    rough = nb.add(nb.mapr(crease, 0.3, 0.7, 0.3, 0.5), nb.mul(wear, 0.3))
    h = nb.add(nb.mul(pores, 0.3), nb.mul(nb.mapr(crease, 0.55, 0.8), 0.7))
    met = 0.0
    if extra:
        c, rough, met, h = extra(nb, c, rough, met, h)
    return nb.finish(c, rough, met, nb.bump(h, 0.3, 0.003))


def _webbing(nb):
    """Black nylon strap webbing."""
    co = nb.coord('Object')
    weave = nb.noise(co, 160.0, 2, 0.5)
    mott = nb.noise(co, 30.0, 3, 0.5)
    open_ = nb.mapr(nb.ao(0.03), 0.35, 0.95)
    c = nb.mixc(nb.mapr(nb.add(nb.mul(weave, 0.6), nb.mul(mott, 0.4)), 0.3, 0.75), (0.008, 0.008, 0.009),
                (0.026, 0.025, 0.026))
    c = nb.mixc(nb.mul(nb.mul(edge_mask(nb, 0.003, 0.03, 0.25), open_), 0.5), c, (0.06, 0.058, 0.058))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.8), c, (0.002, 0.002, 0.002))
    return nb.finish(c, nb.mapr(weave, 0.3, 0.7, 0.7, 0.9), 0.0, nb.bump(weave, 0.35, 0.002))


RECIPES = {'silver': _silver, 'pewter': _pewter, 'iron': _iron, 'branch': _branch, 'leaf': _leaf,
           'gem': _gem, 'wood': _wood, 'string': _string, 'chain_iron': lambda nb: _iron(nb, extra=rust),
           'leather': _leather, 'webbing': _webbing}


def material(name):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    RECIPES[name](NB(mat))
    return mat


# ---------------------------------------------------------------------------------------- bake
def _rewire(mat, mode):
    n, links = mat.node_tree.nodes, mat.node_tree.links
    out, em = n['OUT'], n['BAKE_EMIT']
    for s in (out.inputs['Surface'], em.inputs['Color']):
        for lk in list(s.links):
            links.remove(lk)
    if mode == 'BSDF':
        links.new(n['BSDF'].outputs['BSDF'], out.inputs['Surface'])
        return
    src = _sock(n['BAKE_ALB'].outputs, 'Result_Color') if mode == 'ALB' else n['BAKE_RM'].outputs[0]
    links.new(src, em.inputs['Color'])
    links.new(em.outputs['Emission'], out.inputs['Surface'])


def _pixels(img):
    w, h = img.size
    a = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(h, w, 4)[::-1]


def _to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _down(a, f):
    h, w = a.shape[:2]
    return a.reshape(h // f, f, w // f, f, *a.shape[2:]).mean((1, 3))


def bake(ob, outdir, base, samples=16, size=COLOR_SIZE, post=None):
    """Bake the procedural materials into Roblox-ready maps. Returns paths for ALB/NOR/RGH/MTL."""
    from PIL import Image
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.render.bake.margin = 16
    sc.render.bake.use_clear = True
    select_only(ob)
    mats = [s.material for s in ob.material_slots]
    res = {}
    for key, mode, btype, spp in (('ALB', 'ALB', 'EMIT', samples), ('RM', 'RM', 'EMIT', samples),
                                  ('NOR', 'BSDF', 'NORMAL', 4)):
        img = bpy.data.images.new(f'{base}_{key}_bake', size, size, alpha=False, float_buffer=True, is_data=True)
        for m in mats:
            _rewire(m, mode)
            tgt = m.node_tree.nodes.get('BAKE_TGT') or m.node_tree.nodes.new('ShaderNodeTexImage')
            tgt.name = 'BAKE_TGT'
            tgt.image = img
            m.node_tree.nodes.active = tgt
        sc.cycles.samples = spp
        bpy.ops.object.bake(type=btype, normal_space='TANGENT', margin=16, use_clear=True)
        res[key] = _pixels(img)[..., :3].copy()
        bpy.data.images.remove(img)
    for m in mats:
        _rewire(m, 'BSDF')
        m.node_tree.nodes.remove(m.node_tree.nodes['BAKE_TGT'])
    os.makedirs(outdir, exist_ok=True)
    alb = res['ALB']
    if post:
        alb = post(alb)
    f = size // PBR_SIZE
    nor = _down(res['NOR'] * 2.0 - 1.0, f)
    nor = nor / np.maximum(np.linalg.norm(nor, axis=-1, keepdims=True), 1e-6) * 0.5 + 0.5
    rm = _down(res['RM'], f)
    paths = {k: os.path.join(outdir, f'{base}_{k}.png') for k in ('ALB', 'NOR', 'RGH', 'MTL')}
    Image.fromarray((_to_srgb(alb) * 255 + 0.5).astype(np.uint8), 'RGB').save(paths['ALB'])
    Image.fromarray((np.clip(nor, 0, 1) * 255 + 0.5).astype(np.uint8), 'RGB').save(paths['NOR'])
    Image.fromarray((np.clip(rm[..., 0], 0, 1) * 255 + 0.5).astype(np.uint8), 'L').save(paths['RGH'])
    Image.fromarray((np.clip(rm[..., 1], 0, 1) * 255 + 0.5).astype(np.uint8), 'L').save(paths['MTL'])
    return paths


def final_material(ob, tex, name):
    mat = bpy.data.materials.new(name)
    nb = NB(mat)

    def img(path, data):
        im = bpy.data.images.load(path, check_existing=True)
        im.colorspace_settings.name = 'Non-Color' if data else 'sRGB'
        return im

    uv = nb.coord('UV')
    col = nb.image(img(tex['ALB'], False), uv, 'EXTEND')
    nm = nb.node('ShaderNodeNormalMap')
    nb.feed(nm.inputs['Color'], nb.image(img(tex['NOR'], True), uv, 'EXTEND'))
    bsdf = nb.node('ShaderNodeBsdfPrincipled', name='BSDF')
    nb.feed(bsdf.inputs['Base Color'], col)
    nb.feed(bsdf.inputs['Roughness'], nb.image(img(tex['RGH'], True), uv, 'EXTEND'))
    nb.feed(bsdf.inputs['Metallic'], nb.image(img(tex['MTL'], True), uv, 'EXTEND'))
    nb.feed(bsdf.inputs['Normal'], nm.outputs['Normal'])
    out = nb.node('ShaderNodeOutputMaterial', name='OUT')
    nb.L.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    ob.data.polygons.foreach_set('material_index', [0] * len(ob.data.polygons))
    for m in [m for m in bpy.data.materials if m != mat and m.users == 0]:
        bpy.data.materials.remove(m)
    return mat


# ---------------------------------------------------------------------------- validate + export
def mesh_stats(ob):
    me = ob.data
    v = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', v)
    v = v.reshape(-1, 3) @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]
    bm = bmesh.new()
    bm.from_mesh(me)
    open_edges = sum(1 for e in bm.edges if len(e.link_faces) != 2)
    tiny = sum(1 for f in bm.faces if f.calc_area() < 1e-9)
    bm.free()
    return {'tris': sum(len(p.vertices) - 2 for p in me.polygons), 'min': v.min(0), 'max': v.max(0),
            'open_edges': open_edges, 'degenerate_faces': tiny, 'materials': len(me.materials),
            'uv_layers': len(me.uv_layers)}


def validate(ob, kind, attach):
    st = mesh_stats(ob)
    a = np.array(ATTACH[attach])
    lo, hi = st['min'] - a, st['max'] - a
    hx, yf, yb, zd, zu = LIMITS[kind]
    bounds_ok = bool(max(abs(lo[0]), abs(hi[0])) <= hx and -lo[1] <= yf and hi[1] <= yb
                     and -lo[2] <= zd and hi[2] <= zu)
    rep = {'tris': st['tris'], 'tri_ok': st['tris'] <= TRI_LIMIT, 'watertight': st['open_edges'] == 0,
           'degenerate_faces': st['degenerate_faces'], 'materials': st['materials'], 'uv_layers': st['uv_layers'],
           'size_studs': [round(float(x), 3) for x in st['max'] - st['min']],
           'rel_min': [round(float(x), 3) for x in lo], 'rel_max': [round(float(x), 3) for x in hi],
           'bounds_ok': bounds_ok, 'type': kind, 'attachment': attach}
    print('VALIDATION', json.dumps(rep))
    return rep


def export_item(ob, outdir, base, kind, attach, meta=None):
    """Validate on the mannequin, recentre to the origin, save .blend + FBX + metadata."""
    rep = validate(ob, kind, attach)
    st = mesh_stats(ob)
    center = (st['min'] + st['max']) / 2
    ob.data.transform(Matrix.Translation(Vector(-center)))
    ob.location = (0, 0, 0)
    rep['fit_center'] = [round(float(x), 4) for x in center]
    off = center - np.array(ATTACH[attach])
    rep['offset_from_attachment'] = [round(float(x), 4) for x in off]
    # Position of the Handle's attachment in the Handle's own space, Roblox axes (x right, y up, z back).
    rep['roblox_attachment'] = attach + 'Attachment'
    rep['roblox_handle_attachment_position'] = [round(float(x), 4) + 0.0 for x in (off[0], -off[2], -off[1])]
    rep['rig'] = RIG
    if meta:
        rep.update(meta)
    select_only(ob)
    fbx = os.path.join(outdir, f'{base}.fbx')
    bpy.ops.export_scene.fbx(
        filepath=fbx, check_existing=False, use_selection=True, object_types={'MESH'},
        use_mesh_modifiers=True, mesh_smooth_type='FACE', use_triangles=True, use_tspace=False,
        use_custom_props=False, colors_type='NONE', add_leaf_bones=False, bake_anim=False,
        apply_unit_scale=True, apply_scale_options='FBX_SCALE_UNITS', global_scale=1.0,
        axis_forward='-Z', axis_up='Y', bake_space_transform=False, path_mode='COPY', embed_textures=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(outdir, f'{base}.blend'), compress=True, relative_remap=True)
    with open(os.path.join(outdir, f'{base}.json'), 'w') as fh:
        json.dump(rep, fh, indent=1)
    return rep


def dev_modes(asm, name, views):
    """Triangle tally and body clearance (returned for the report). DRY=1 exits before the bake;
    LOOK=1 first renders the un-baked procedural materials to .cache."""
    v = np.concatenate(asm.v)
    hits, gap = body_check(v, [g for g, x in zip(asm.part_group, asm.v) for _ in range(len(x))]
                           if len(asm.part_group) == len(asm.v) else None)
    print('TRIS by group', asm.groups, 'total', asm.tris())
    print('BODY inside:', hits or 'none', f'min clearance {gap:.4f}')
    print('BOUNDS', np.round(v.min(0), 3), np.round(v.max(0), 3))
    if os.environ.get('LOOK'):
        look(asm, name, views)
    if os.environ.get('DRY') or os.environ.get('LOOK'):
        sys.exit(0)
    return {'body_inside': hits, 'min_clearance': round(float(gap), 4)}


def weld(ob, dist=1e-4):
    """Merge coincident vertices and drop the resulting degenerate faces (keeps the mesh closed)."""
    select_only(ob)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=dist)
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    bpy.ops.mesh.fill_holes(sides=0)      # cap every boundary loop, whatever its size
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    st = mesh_stats(ob)
    print(f'WELD tris={st["tris"]} open_edges={st["open_edges"]}')


def finish_item(asm, name, kind, attach, display, bake_spp=16, meta=None):
    """Build -> pack UVs -> bake -> final material -> validate/export. Returns (object, report)."""
    out = os.path.join(ASSETS, name)
    print('TRIS', asm.tris())
    ob = asm.build(name)
    weld(ob)
    pack_uv(ob)
    tex = bake(ob, os.path.join(out, 'textures'), name, samples=int(os.environ.get('BAKE_SPP', bake_spp)))
    final_material(ob, tex, name + '_Mat')
    rep = export_item(ob, out, name, kind, attach, {'display_name': display, **(meta or {})})
    ob.location = rep['fit_center']
    return ob, rep


# ----------------------------------------------------------------------------------- previews
def studio(target, cam_dir, dist, lens=60.0, res=(900, 900), samples=48, bg=(0.62, 0.62, 0.64), exposure=-0.35):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    try:
        sc.view_settings.view_transform = 'AgX'
        sc.view_settings.look = 'AgX - Medium High Contrast'
    except TypeError:
        pass
    sc.view_settings.exposure = exposure
    w = sc.world
    w.color = bg
    if w.node_tree is None:
        w.use_nodes = True
    bgn = w.node_tree.nodes.get('Background') or w.node_tree.nodes.new('ShaderNodeBackground')
    bgn.inputs['Color'].default_value = (*bg, 1)
    bgn.inputs['Strength'].default_value = 0.7
    outn = w.node_tree.nodes.get('World Output') or w.node_tree.nodes.new('ShaderNodeOutputWorld')
    w.node_tree.links.new(bgn.outputs[0], outn.inputs['Surface'])
    t = Vector(target)
    d = Vector(cam_dir).normalized()
    cam = bpy.data.objects.new('Cam', bpy.data.cameras.new('Cam'))
    cam.data.lens = lens
    cam.data.clip_start = 0.01
    sc.collection.objects.link(cam)
    cam.location = t + d * dist
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    sc.camera = cam
    for nm, off, power, size in (('Key', (-0.8, -1.0, 0.9), 900, 3.0), ('Fill', (1.0, -0.6, 0.2), 250, 4.0),
                                 ('Rim', (0.4, 1.2, 1.0), 700, 2.0)):
        if d.y > 0:                     # camera behind the avatar: light it from that side
            off = (off[0], -off[1], off[2])
        ld = bpy.data.lights.new(nm, 'AREA')
        ld.energy = power * dist * dist / 36.0
        ld.size = size * dist / 6.0
        lo = bpy.data.objects.new(nm, ld)
        sc.collection.objects.link(lo)
        lo.location = t + Vector(off).normalized() * dist
        lo.rotation_euler = (t - lo.location).to_track_quat('-Z', 'Y').to_euler()
    return cam


def render(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def clear_studio():
    for nm in ('Cam', 'Key', 'Fill', 'Rim'):
        o = bpy.data.objects.get(nm)
        if o is not None:
            bpy.data.objects.remove(o)


def previews(name, views, folder=None, spp=40, res=(900, 900)):
    """Render views [(suffix, target, cam_dir, dist, lens)] with the mannequin into `folder`."""
    load_mannequin()
    folder = folder or os.path.join(ASSETS, name)
    out = []
    for suffix, target, cam_dir, dist, lens in views:
        clear_studio()
        studio(target, cam_dir, dist, lens=lens, res=res, samples=spp)
        out.append(os.path.join(folder, f'{name}_{suffix}.png'))
        render(out[-1])
    return out


def look(asm, name, views):
    """Quick un-baked render of the procedural materials, for iterating on shape."""
    asm.build(name)
    os.makedirs(CACHE, exist_ok=True)
    return previews(name, views, CACHE, spp=int(os.environ.get('LOOK_SPP', 12)), res=(640, 640))
