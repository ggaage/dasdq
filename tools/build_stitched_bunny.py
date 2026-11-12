"""Ashen Stitched Bunny: Shoulder accessory (LeftCollarAttachment), fitted to the default R6 avatar.

The collection's moe piece: a worn, hand-repaired plush rabbit in dusty ash-beige velour sits on the
wearer's left shoulder and rests its too-big head against theirs. One glossy black button eye sewn on
with oxblood thread and one cross-stitched X eye, a sewn-shut seam mouth, a pinked faded-rose heart patch
held on with big running stitches, a side seam split open with stuffing escaping past one surviving
stitch, and one ear upright while the other is broken, flopped over and bound in old gauze with a safety
pin through it. The plush body is a signed-distance model cut against the R6 torso and head, so it
squashes softly where it sits and where it leans. One closed mesh, < 4k triangles.
"""
import math
import os
import sys

import bpy  # noqa: F401  (must precede mathutils)
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbxkit as K  # noqa: E402

K.use_rig('R6')
import rbxsdf as S  # noqa: E402

NAME = 'AshenStitchedBunny'
np.random.seed(52)                 # Asm draws per-part shading noise from the global generator
rng = np.random.default_rng(52)
X, Y, Z = np.eye(3)
Z0 = K.ATTACH['LeftCollar'][2]     # top of the R6 torso / arm: the plush sits on it
HEAD_SIDE = 0.595                  # flat side of the R6 head (x) between z 2.2 and 2.85
C_SEAT, C_HEAD = 0.012, 0.02       # clearance kept from the shoulder top and from the head

# ------------------------------------------------------------------------------------ materials
ASH_LO, ASH_HI = (0.150, 0.128, 0.100), (0.315, 0.282, 0.236)
ASH_GREY, ASH_GRIME, ASH_WORN = (0.20, 0.19, 0.175), (0.075, 0.062, 0.05), (0.36, 0.335, 0.30)
ASH_SEAM, ASH_CREV = (0.05, 0.042, 0.034), (0.03, 0.024, 0.019)


def _plush(nb):
    """Dusty ash-beige velour: brushed pile, pilling, grime collected low down, sewn seams (vfac = signed
    distance to the seam line, 0 on it)."""
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    z = nb.xyz(co)[2]
    fib = nb.noise(co, 240.0, 2, 0.5)
    nap = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 3.0)), 55.0, 4, 0.6, distortion=0.6)
    mott = nb.noise(co, 7.0, 4, 0.55)
    pills = nb.mul(nb.mapr(nb.voronoi(co, 95.0), 0.2, 0.06), nb.mapr(nb.noise(co, 11.0, 2, 0.5), 0.5, 0.62))
    open_ = nb.mapr(nb.ao(0.06), 0.25, 0.95)
    seam = nb.mapr(nb.absf(nb.attr('vfac')), 0.0042, 0.0012)
    worn = nb.mul(K.edge_mask(nb, 0.012, 0.02, 0.2), open_)
    c = nb.mixc(nb.mapr(nb.add(nb.mul(nap, 0.55), nb.mul(fib, 0.45)), 0.28, 0.78), ASH_LO, ASH_HI)
    c = nb.mixc(nb.mul(nb.mapr(nb.add(mott, nb.mul(rnd, 0.2)), 0.4, 0.75), 0.5), c, ASH_GREY)
    low = nb.mapr(z, Z0 + 0.14, Z0 + 0.015)
    grime = nb.mapr(nb.add(nb.mul(low, 0.65), nb.mul(nb.noise(co, 16.0, 4, 0.6), 0.7)), 0.5, 1.05)
    c = nb.mixc(nb.mul(grime, 0.55), c, ASH_GRIME)
    c = nb.mixc(nb.mul(pills, 0.45), c, ASH_HI)
    c = nb.mixc(nb.mul(worn, 0.3), c, ASH_WORN)
    c = nb.mixc(nb.mul(seam, 0.8), c, ASH_SEAM)
    c = nb.mixc(nb.mul(nb.inv(open_), 0.75), c, ASH_CREV)
    rough = nb.sub(nb.mapr(fib, 0.3, 0.7, 0.86, 0.97), nb.mul(worn, 0.12))
    h = nb.add(nb.add(nb.mul(fib, 0.25), nb.mul(nap, 0.35)), nb.sub(nb.mul(pills, 0.4), nb.mul(seam, 0.9)))
    return nb.finish(c, rough, 0.0, nb.bump(h, 0.35, 0.003))


def felt(lo, hi, faded, grime):
    """Matted felt with sun-faded blotches and fuzzy cut edges."""
    def recipe(nb):
        co = nb.coord('Object')
        rnd = nb.attr('part_rnd')
        fib = nb.noise(co, 300.0, 2, 0.5)
        clump = nb.noise(co, 70.0, 4, 0.6, distortion=0.8)
        fade = nb.noise(co, 10.0, 3, 0.5)
        open_ = nb.mapr(nb.ao(0.03), 0.3, 0.95)
        edge = nb.mul(K.edge_mask(nb, 0.003, 0.03, 0.25), open_)
        c = nb.mixc(nb.mapr(nb.add(nb.mul(clump, 0.6), nb.mul(fib, 0.4)), 0.3, 0.75), lo, hi)
        c = nb.mixc(nb.mul(nb.mapr(nb.add(fade, nb.mul(rnd, 0.3)), 0.45, 0.8), 0.7), c, faded)
        c = nb.mixc(nb.mul(edge, 0.4), c, faded)
        c = nb.mixc(nb.mul(nb.inv(open_), 0.7), c, grime)
        h = nb.add(nb.mul(clump, 0.6), nb.mul(fib, 0.4))
        return nb.finish(c, nb.mapr(fib, 0.3, 0.7, 0.88, 0.97), 0.0, nb.bump(h, 0.3, 0.002))
    return recipe


def _thread(nb):
    """Waxed oxblood sewing thread."""
    co = nb.coord('Object')
    ply = nb.noise(co, 420.0, 2, 0.5)
    n1 = nb.noise(co, 60.0, 3, 0.5)
    open_ = nb.mapr(nb.ao(0.01), 0.3, 0.95)
    c = nb.mixc(nb.mapr(nb.add(nb.mul(ply, 0.5), nb.mul(n1, 0.5)), 0.3, 0.75), (0.055, 0.008, 0.009),
                (0.12, 0.022, 0.02))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.7), c, (0.012, 0.003, 0.003))
    return nb.finish(c, nb.mapr(ply, 0.3, 0.7, 0.5, 0.72), 0.0, nb.bump(ply, 0.4, 0.0015))


def _button(nb):
    """Glossy black button with fine scratches and lint caught round the thread."""
    co = nb.coord('Object')
    n1 = nb.noise(co, 40.0, 3, 0.5)
    scr = nb.mapr(nb.noise(nb.mapping(co, scale=(1.0, 8.0, 1.0), rot=(0.3, 0.5, 0.9)), 90.0, 2, 0.5), 0.62, 0.7)
    open_ = nb.mapr(nb.ao(0.012), 0.3, 0.95)
    c = nb.mixc(nb.mapr(n1, 0.35, 0.7), (0.006, 0.0055, 0.0055), (0.017, 0.015, 0.014))
    c = nb.mixc(nb.mul(scr, 0.5), c, (0.05, 0.047, 0.044))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.8), c, (0.045, 0.04, 0.035))
    rough = nb.add(nb.mapr(n1, 0.3, 0.7, 0.1, 0.2), nb.mul(scr, 0.35))
    return nb.finish(c, rough, 0.0, nb.bump(nb.mul(scr, -1.0), 0.15, 0.001))


def _gauze(nb):
    """Old cotton gauze: open weave, yellowed, a faded rust-brown stain."""
    co = nb.coord('Object')
    x, y, z = nb.xyz(co)
    w1 = nb.absf(nb.math('SINE', nb.mul(nb.add(x, z), 760.0)))
    w2 = nb.absf(nb.math('SINE', nb.mul(nb.sub(y, z), 760.0)))
    weave = nb.maxf(w1, w2)
    n1 = nb.noise(co, 30.0, 4, 0.6)
    stain = nb.mapr(nb.noise(co, 14.0, 4, 0.65, distortion=0.8), 0.6, 0.75)
    open_ = nb.mapr(nb.ao(0.02), 0.3, 0.95)
    c = nb.mixc(nb.mapr(nb.add(nb.mul(weave, 0.4), nb.mul(n1, 0.6)), 0.35, 0.85), (0.42, 0.39, 0.33),
                (0.56, 0.535, 0.47))
    c = nb.mixc(nb.mul(nb.mapr(n1, 0.5, 0.8), 0.6), c, (0.40, 0.33, 0.20))
    c = nb.mixc(nb.mul(stain, 0.55), c, (0.13, 0.03, 0.025))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.75), c, (0.18, 0.16, 0.13))
    return nb.finish(c, nb.mapr(weave, 0.3, 1.0, 0.92, 0.8), 0.0, nb.bump(weave, 0.3, 0.0015))


def _stuffing(nb):
    """Polyester fibrefill, greyed with dust in its folds."""
    co = nb.coord('Object')
    fluff = nb.noise(co, 140.0, 5, 0.65, distortion=1.5)
    n1 = nb.noise(co, 35.0, 3, 0.5)
    open_ = nb.mapr(nb.ao(0.02), 0.25, 0.95)
    c = nb.mixc(nb.mapr(nb.add(nb.mul(fluff, 0.6), nb.mul(n1, 0.4)), 0.3, 0.8), (0.50, 0.49, 0.47),
                (0.64, 0.63, 0.60))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.8), c, (0.22, 0.20, 0.18))
    return nb.finish(c, 0.95, 0.0, nb.bump(fluff, 0.6, 0.003))


K.RECIPES.update({
    'plush': _plush, 'thread': _thread, 'button': _button, 'gauze': _gauze, 'stuffing': _stuffing,
    'felt_patch': felt((0.20, 0.075, 0.07), (0.34, 0.15, 0.135), (0.33, 0.22, 0.20), (0.09, 0.05, 0.045)),
    'felt_ear': felt((0.30, 0.18, 0.16), (0.42, 0.28, 0.25), (0.40, 0.32, 0.29), (0.14, 0.10, 0.085))})


# --------------------------------------------------------------------------------- math helpers
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot3(axis, deg):
    return np.array(Matrix.Rotation(math.radians(deg), 3, axis))


def rot_about(v, axis, deg):
    return np.array(Matrix.Rotation(math.radians(deg), 3, Vector(axis)) @ Vector(v))


def m4(R=None, t=(0.0, 0.0, 0.0)):
    m = np.eye(4)
    if R is not None:
        m[:3, :3] = R
    m[:3, 3] = t
    return m


def tp(m, p):
    p = np.asarray(p, float)
    return p @ m[:3, :3].T + m[:3, 3]


def tv(m, v):
    return np.asarray(v, float) @ m[:3, :3].T


def resample(p, n):
    """n points evenly spaced along the polyline p; also their normalised arc length and the length."""
    p = np.asarray(p, float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, p[:, k]) for k in range(3)], 1), u / s[-1], s[-1]


def at_s(arr, s, q):
    arr = np.asarray(arr, float)
    return np.interp(q, s, arr) if arr.ndim == 1 else np.array([np.interp(q, s, arr[:, k]) for k in range(3)])


# ------------------------------------------------------------------------------- the plush (SDF)
# Bunny space: origin at the seat, facing -Y, +X = the bunny's (and wearer's) left, Z up.
YAW, ROLL, PITCH = 18.0, -6.0, -4.0             # face a little outwards, lean onto the head, sit back
R_B = rot3('Z', YAW) @ rot3('Y', ROLL) @ rot3('X', PITCH)
NECK = np.array([0.0, 0.0, 0.265])
R_H = rot3('Y', -7.0) @ rot3('X', 6.0)          # head tipped onto the wearer, drooping forwards
M_H = m4(None, NECK) @ m4(R_H) @ m4(None, -NECK)
HC = np.array([0.0, -0.01, 0.43])               # head centre before the tilt


def head_sdf(q):
    d = S.ellipsoid(q, HC, (0.225, 0.19, 0.198))
    for s in (-1, 1):                                                            # puffy cheeks
        d = S.smin(d, S.ellipsoid(q, HC + (s * 0.105, -0.075, -0.05), (0.1, 0.09, 0.088)), 0.05)
    return S.smin(d, S.ellipsoid(q, HC + (0.0, -0.162, -0.065), (0.066, 0.046, 0.048)), 0.035)   # muzzle


def body_sdf(p):
    d = S.ellipsoid(p, (0.0, 0.0, 0.13), (0.152, 0.135, 0.14))
    d = S.smin(d, S.ellipsoid(p, (0.0, 0.008, 0.235), (0.11, 0.098, 0.085)), 0.06)
    for s in (-1, 1):                                                            # stubby legs, round feet
        d = S.smin(d, S.capsule(p, (s * 0.068, -0.035, 0.068), (s * 0.1, -0.165, 0.058), 0.06, 0.055), 0.03)
        d = S.smin(d, S.ellipsoid(p, (s * 0.104, -0.2, 0.06), (0.058, 0.07, 0.056)), 0.025)
    d = S.smin(d, S.capsule(p, (-0.112, -0.02, 0.228), (-0.138, -0.12, 0.112), 0.048, 0.045), 0.022)  # inner arm
    d = S.smin(d, S.capsule(p, (0.118, 0.005, 0.228), (0.158, 0.03, 0.1), 0.048, 0.045), 0.022)      # outer arm
    return S.smin(d, S.sphere(p, (0.0, 0.14, 0.085), 0.05), 0.02)                                     # tail


def plush(p):
    return S.smin(body_sdf(p), head_sdf(S.rotated(p, NECK, R_H)), 0.035)


def project(fn, p, iters=10, h=1e-4):
    """Closest point of fn's zero set (Newton steps along the gradient) and the outward normal there."""
    p = np.asarray(p, float).copy()
    for _ in range(iters + 1):
        g = np.array([(fn((p + h * e)[None])[0] - fn((p - h * e)[None])[0]) / (2 * h) for e in np.eye(3)])
        p = p - fn(p[None])[0] * g / max(g @ g, 1e-12)
    return p, unit(g)


K.ensure_mannequin()
K.new_scene()
hob = K.load_mannequin(['Head'])[0]
hm = np.array(hob.matrix_world)
HEAD_BVH = BVHTree.FromPolygons(tp(hm, [v.co[:] for v in hob.data.vertices]).tolist(),
                                [list(f.vertices) for f in hob.data.polygons])
bpy.data.objects.remove(hob)

# Seat: lowest plush point 0.03 below the shoulder plane (the cut flattens it there); the head's inner
# cheek just reaches the side of the wearer's head (the cut squashes it by the clearance).
g = np.stack(np.meshgrid(np.arange(-0.32, 0.32, 0.008), np.arange(-0.36, 0.26, 0.008),
                         np.arange(-0.08, 0.72, 0.008), indexing='ij'), -1).reshape(-1, 3)
w_all = g[plush(g) < 0] @ R_B.T
w_head = g[head_sdf(S.rotated(g, NECK, R_H)) < 0] @ R_B.T
SEAT = np.array([HEAD_SIDE - 0.004 - w_head[:, 0].min(), -0.08, Z0 + C_SEAT - 0.03 - w_all[:, 2].min()])
M_B = m4(R_B, SEAT)
LO_B, HI_B = g[plush(g) < 0].min(0) - 0.02, g[plush(g) < 0].max(0) + 0.02
print('SEAT', SEAT.round(3), 'head contact z', round(float(w_head[w_head[:, 0].argmin(), 2] + SEAT[2]), 3),
      'plush box', LO_B.round(3), HI_B.round(3))
del g

# Split side seam: a lens-shaped opening on the outer flank between arm and thigh.
SLIT_C, e3 = project(plush, (0.19 * math.sin(math.radians(70)), -0.19 * math.cos(math.radians(70)), 0.16))
e1 = rot_about(unit(Z - e3 * (Z @ e3)), e3, 8.0)
SLIT_R = np.column_stack([e1, np.cross(e3, e1), e3])
e2 = SLIT_R[:, 1]


def head_dist(w):
    """Signed distance to the R6 head mesh (positive outside), +1 beyond 0.25 studs."""
    flat = w.reshape(-1, 3)
    out = np.ones(len(flat))
    fn = HEAD_BVH.find_nearest
    for i in np.nonzero((flat[:, 0] < 0.9) & (flat[:, 2] > 1.8))[0]:
        p = Vector(flat[i])
        loc, nrm, _, dist = fn(p, 0.25)
        if loc is not None:
            out[i] = dist if (p - loc).dot(nrm) >= 0 else -dist
    return out.reshape(w.shape[:-1])


def plush_fitted(p):
    d = S.cut(plush(p), S.ellipsoid(p, SLIT_C - e3 * 0.006, (0.05, 0.0135, 0.026), SLIT_R), 0.006)
    w = tp(M_B, p)
    body = np.minimum(S.box(w, (0, 0, 1), (1, 0.5, 1)), S.box(w, (1.5, 0, 1), (0.5, 0.5, 1)))
    d = S.cut(d, body - C_SEAT, 0.025)
    return S.cut(d, head_dist(w) - C_HEAD, 0.03)


STUFF = [(-0.034, 0.002, 0.0, 0.013), (-0.016, -0.003, 0.006, 0.017), (0.004, 0.003, 0.011, 0.02),
         (0.022, -0.002, 0.007, 0.017), (0.038, 0.002, -0.002, 0.012)]   # (e1, e2, e3, radius)


def stuffing_sdf(p):
    q = (p - SLIT_C) @ SLIT_R
    d = None
    for a, b, c, r in STUFF:
        s = np.linalg.norm(q - (a, b, c), axis=-1) - r
        d = s if d is None else S.smin(d, s, 0.009)
    return d + 0.0035 * S.waves(p, 160.0, seed=5, octaves=3)


BV_V, BV_F, BV_UV = S.mesh(plush_fitted, LO_B, HI_B, 0.009, 2000, name='plush')
ST = S.mesh(stuffing_sdf, SLIT_C - 0.068, SLIT_C + 0.068, 0.004, 170, name='stuffing')
BV = BVHTree.FromPolygons(BV_V.tolist(), BV_F)


def seam_field(v):
    """Signed distance to the plush's centre seam (head and body are sewn from left/right halves)."""
    hv = S.rotated(v, NECK, R_H)
    return np.where(head_sdf(hv) < body_sdf(v), hv[:, 0], v[:, 0])


A = K.Asm()
A.part('plush', (BV_V, BV_F, BV_UV), 'plush', vfac=seam_field(BV_V))
A.part('stuffing', ST, 'stuffing')


# --------------------------------------------------------------------------------- surface tools
def hit(o, d):
    loc, nrm, _, _ = BV.ray_cast(Vector(o), Vector(d))
    return np.array(loc), np.array(nrm)


def near(p):
    loc, nrm, _, _ = BV.find_nearest(Vector(p))
    return np.array(loc), np.array(nrm)


HX, HY, HZ = tv(M_H, X), tv(M_H, Y), tv(M_H, Z)
HCB = tp(M_H, HC)


def on_head(az, el):
    """Point and normal on the face along head direction (azimuth from the front towards +X, elevation)."""
    d = math.sin(az) * math.cos(el) * HX - math.cos(az) * math.cos(el) * HY + math.sin(el) * HZ
    return hit(HCB + d * 0.6, -d)


def tangent_frame(n, up=Z):
    u = unit(up - n * (up @ n))
    return np.cross(u, n), u          # right, up (right x up = n)


def stitch(pts, r, lift=0.6, sink=0.8, base=0.0, group='threads'):
    """Thread through approximate points: snapped to the cloth (+ base, e.g. a patch's thickness); the
    inner points ride above it, the ends dive into it."""
    out, n = [], len(pts)
    for i, p in enumerate(pts):
        loc, nrm = near(p)
        end = i in (0, n - 1)
        out.append(loc + nrm * (base + (-sink * r if end else lift * r * (0.6 + 0.4 * math.sin(math.pi * i / (n - 1))))))
    A.part(group, K.tube(out, r, sides=3), 'thread', smooth=70)


# --------------------------------------------------------------------------------------- face
# Button eye on the bunny's left: domed, two holes, two parallel loops of thread.
BUTTON = [(0.0, 0.0175), (0.011, 0.0168), (0.02, 0.0145), (0.027, 0.0105), (0.0318, 0.0055), (0.0336, 0.0),
          (0.031, -0.0045), (0.0, -0.006)]
pb, nb_ = on_head(0.52, -0.06)
nrm_b = unit(nb_ + HZ * 0.05)
_, upb = tangent_frame(nrm_b, HZ)
mb = K.mat_trs(pb - nrm_b * 0.0015, (tuple(nrm_b), tuple(rot_about(upb, nrm_b, 16.0))))
A.part('eyes', K.lathe(BUTTON, segs=12), 'button', mb, smooth=50)
for yy in (-0.0052, 0.0052):
    A.part('eyes', K.tube([K.xf(mb, (x, yy, zz)) for x, zz in ((-0.0125, 0.0132), (-0.0062, 0.0193),
                                                                   (0.0062, 0.0193), (0.0125, 0.0132))],
                          0.0032, sides=3), 'thread', smooth=70)

# Cross-stitched X eye on the right, one stroke a little longer and steeper (sewn by hand).
pe, ne = on_head(-0.52, -0.06)
rt, up = tangent_frame(ne, HZ)
for ang, half in ((38.0, 0.038), (-50.0, 0.034)):
    dv = math.cos(math.radians(ang)) * rt + math.sin(math.radians(ang)) * up
    stitch([pe + dv * half * t for t in np.linspace(-1, 1, 5)], 0.0065, 0.55)

# Mouth sewn shut: a seam across the muzzle crossed by four short stitches.
MOUTH = [(a, -0.43 - 0.06 * (a / 0.17) ** 2) for a in np.linspace(-0.17, 0.17, 7)]
stitch([on_head(a, e)[0] for a, e in MOUTH], 0.0036, 0.3, 0.6)
for a in (-0.12, -0.04, 0.04, 0.12):
    c, n = on_head(a, -0.43 - 0.06 * (a / 0.17) ** 2)
    _, u = tangent_frame(n, HZ)
    stitch([c - u * 0.012, c, c + u * 0.012], 0.0042, 0.7)


# --------------------------------------------------------------------------------------- ears
def ear(ctrl, side_axis, n=12, wmax=0.056, tmax=0.022, pinch=None):
    """Flat plush ear along a spline (width along side_axis, felt panel on the front face). Returns the
    centre line, its normalised arc length, widths and frames for anything wrapped round it."""
    p, s, _ = resample(K.spline(ctrl, 10), n)
    base = 0.6 + 0.4 * np.clip(s / 0.3, 0.0, 1.0) ** 0.7                    # pinched where it is sewn in
    tip = np.sqrt(np.clip(1.0 - np.clip((s - 0.74) / 0.26, 0.0, 1.0) ** 2, 0.0, 1.0))
    w = wmax * base * (1.0 - 0.12 * s) * tip
    t = tmax * (0.85 + 0.15 * base) * tip
    if pinch is not None:                                                    # squeezed by a bandage
        gk = np.exp(-((s - pinch[0]) / pinch[1]) ** 2)
        w, t = w * (1.0 - pinch[2] * gk), t * (1.0 - 0.4 * pinch[2] * gk)
    sides = 8
    rim = np.r_[np.concatenate([np.sin(np.arange(sides) * 2 * np.pi / sides) * t[i] for i in range(n - 1)]), 0.0]
    A.part('ears', K.tube(p, w, sides=sides, caps=('flat', 'point'), up=side_axis, ry=t), 'plush', vfac=rim)
    T, N, B = K.frames(p, up=side_axis)
    k = np.arange(1, n - 1)
    A.part('ears', K.tube(p[k] - B[k] * (0.8 * t[k])[:, None], 0.68 * w[k], sides=6, caps=('flat', 'point'),
                          up=side_axis, ry=0.3 * t[k]), 'felt_ear')
    return p, s, w, t, T, N, B


def ear_root(bx, by):
    loc, nrm = hit(HCB + HX * bx + HY * by + HZ * 0.6, -HZ)
    return loc, nrm


# Upright ear on the inner side: stands straight in the world (leaning a little back and away from the
# wearer's head), tip curling forwards.
r0, n0 = ear_root(-0.07, 0.035)
d = R_B.T @ unit((0.10, 0.22, 1.0))
ear([r0 - n0 * 0.028, r0 + d * 0.06, r0 + d * 0.17 - Y * 0.004, r0 + d * 0.27 - Y * 0.012,
     r0 + d * 0.345 - Y * 0.03], HX)

# Broken outer ear: splays out, kinks where the gauze is wound, and hangs limp.
r1, n1 = ear_root(0.085, 0.03)
d1 = unit(HZ * math.cos(math.radians(40)) + HX * math.sin(math.radians(40)))
d2 = R_B.T @ unit((0.36, -0.1, -1.0))
KINK = r1 + d1 * 0.13
FLOP = ear([r1 - n1 * 0.028, r1 + d1 * 0.055, r1 + d1 * 0.105, KINK + d2 * 0.03, KINK + d2 * 0.12,
            KINK + d2 * 0.21 - R_B.T @ Z * 0.01], HX, pinch=(0.46, 0.08, 0.3))

# Gauze: three snug turns over the kink (bunched on the inside of the bend) and a loose tail.
ep, es, ew, et, eT, eN, eB = FLOP
s_k = float(es[np.argmin([np.linalg.norm(q - KINK) for q in ep])])
for q, tilt in ((s_k - 0.075, 9.0), (s_k, -5.0), (s_k + 0.075, 12.0)):
    c, Tn, Nn = at_s(ep, es, q), unit(at_s(eT, es, q)), unit(at_s(eN, es, q))
    a = rot_about(Tn, Nn, tilt)
    bb = np.cross(a, Nn)
    ww, tt, gap = float(np.interp(q, es, ew)), float(np.interp(q, es, et)), 0.0034
    loop = [c + Nn * (ww + gap) * math.cos(f) + bb * (tt + gap) / math.cos(math.radians(tilt)) * math.sin(f)
            for f in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    A.part('bandage', K.tube(loop, 0.0155, sides=4, phase=math.pi / 4, closed=True, up=a, ry=0.0031), 'gauze',
           smooth=60)
    if q > s_k:                                                    # tail end hanging from the last turn
        down = R_B.T @ -Z
        f0 = int(np.argmax([(p - c) @ down for p in loop]))
        p0 = loop[f0]
        tail = [p0 - bb * 0.002, p0 + down * 0.022 + Tn * 0.004, p0 + down * 0.045 + Tn * 0.01 + Nn * 0.006]
        A.part('bandage', K.tube(tail, 0.0155, sides=4, phase=math.pi / 4, up=Tn, ry=0.0031), 'gauze', smooth=60)

# Safety pin through the gauze on the front of the kink.
c, Tn, Nn = at_s(ep, es, s_k), unit(at_s(eT, es, s_k)), unit(at_s(eN, es, s_k))
front = -np.cross(Tn, Nn)
pin_dir = unit(Nn * math.cos(math.radians(28)) + Tn * math.sin(math.radians(28)))
pin_c = c + front * (float(np.interp(s_k, es, et)) + 0.0034 + 0.0022 + 0.0035)
K.safety_pin(A, 'pin', pin_c - pin_dir * 0.066, pin_dir, front, length=0.12, wire=0.0042, mat='silver')


# -------------------------------------------------------------------------------- heart patch
def heart(width, n):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return np.c_[x, y + 2.5] * width / 32


def seg_dist(p, poly):
    a, b = poly, np.roll(poly, -1, 0)
    ab = b - a
    h = np.clip(np.sum((p - a) * ab, 1) / np.sum(ab * ab, 1), 0, 1)
    return float(np.min(np.linalg.norm(p - a - ab * h[:, None], axis=1)))


PW = 0.112
pc, pn = hit((0.6 * math.sin(0.26), -0.6 * math.cos(0.26), 0.172), (-math.sin(0.26), math.cos(0.26), 0.0))
prt, pup = tangent_frame(pn)
prt, pup = rot_about(prt, pn, -13.0), rot_about(pup, pn, -13.0)
smooth_h = heart(PW, 72)
outline = heart(PW, 36)
tng = np.roll(outline, -1, 0) - np.roll(outline, 1, 0)
nrm2 = np.c_[tng[:, 1], -tng[:, 0]] / np.linalg.norm(tng, axis=1)[:, None]
if np.sum(outline[:, 0] * np.roll(outline[:, 1], -1) - np.roll(outline[:, 0], -1) * outline[:, 1]) < 0:
    nrm2 = -nrm2                                                    # outward normals
zig = np.where(np.arange(len(outline)) % 2 == 0, 1.0, -1.0) * 0.0032
zig[[0, 1, -1]] = 0.0                                               # leave the cleft between the lobes clean
outline = outline + nrm2 * zig[:, None]
gx = np.arange(-PW / 2, PW / 2, 0.02)
inner = [(x, y) for x in gx for y in np.arange(-0.06, 0.06, 0.02)
         if K.point_in_poly((x, y), smooth_h) and seg_dist(np.array([x, y]), smooth_h) > 0.011]
v2, tris = K.triangulate_region([tuple(q) for q in outline], (), inner)
PF, PB = 0.0055, 0.004                                              # felt above / below the plush surface
surf = [near(pc + prt * x + pup * y) for x, y in v2]
A.part('patch', K.solid_from_surface(np.array([l + n * PF for l, n in surf]), tris,
                                     np.array([l - n * PB for l, n in surf]), v2), 'felt_patch', smooth=50)

# Big running stitches round the patch.
ins = heart(PW - 0.034, 72)
seg = np.linalg.norm(np.diff(np.vstack([ins, ins[:1]]), axis=0), axis=1)
arc = np.r_[0.0, np.cumsum(seg)]
SL, GAP = 0.024, 0.0135
for k in range(int(arc[-1] // (SL + GAP))):
    q0 = 0.006 + k * (SL + GAP)
    pts2 = [np.array([np.interp(q, arc, np.r_[ins[:, i], ins[0, i]]) for i in (0, 1)]) for q in (q0, q0 + SL / 2, q0 + SL)]
    stitch([pc + prt * x + pup * y for x, y in pts2], 0.0045, 0.8, 1.0, base=PF)

# --------------------------------------------------------------------------- split seam threads
# One surviving stitch strained across the stuffing, and the broken ends at either end of the split.
a = near(SLIT_C - e2 * 0.022 + e1 * 0.01)[0]
b = near(SLIT_C + e2 * 0.022 + e1 * 0.016)[0]
A.part('threads', K.tube([a - e3 * 0.003, a + e3 * 0.012 + (b - a) * 0.15, (a + b) / 2 + e3 * 0.03,
                          b + e3 * 0.012 - (b - a) * 0.15, b - e3 * 0.003], 0.0034, sides=3), 'thread', smooth=70)
for sgn in (-1, 1):
    p = near(SLIT_C + e1 * sgn * 0.05)[0]
    A.part('threads', K.tube([p - e3 * 0.003, p + e3 * 0.01 + e1 * sgn * 0.006,
                              p + e3 * 0.019 + e1 * sgn * 0.013 + e2 * 0.006,
                              p + e3 * 0.024 + e1 * sgn * 0.022 + e2 * 0.013], 0.003, sides=3), 'thread', smooth=70)

A.transform(M_B)

# ------------------------------------------------------------------------------------- output
HEADW = tp(M_B, HCB)
MIDW = tp(M_B, (0.0, -0.02, 0.38))
FACE = R_B @ -Y
VIEWS = [('preview', tuple(MIDW), (1.0, -0.95, 0.38), 2.9, 50),
         ('front', (0.32, 0.0, 2.48), (0.0, -1.0, 0.1), 4.4, 50),
         ('closeup', tuple(HEADW + (0.0, 0.0, 0.03)), (FACE[0] + 0.12, FACE[1], 0.22), 1.5, 50),
         ('side', tuple(MIDW), (1.0, 0.25, 0.22), 2.6, 50)]
print('HEAD centre world', HEADW.round(3), 'mid', MIDW.round(3))

body = K.dev_modes(A, NAME, VIEWS)
ob, rep = K.finish_item(A, NAME, 'Shoulder', 'LeftCollar', 'Ashen Stitched Bunny', meta=body)
K.previews(NAME, VIEWS, spp=int(os.environ.get('PREVIEW_SPP', 40)))
print('DONE', NAME)
