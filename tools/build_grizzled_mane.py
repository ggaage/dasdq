"""Ashen Grizzled Mane: Neck accessory (NeckAttachment), fitted to the default R6 avatar.

A heavy, shaggy wolf's ruff that wraps the base of the head and spills over the shoulders. Charcoal fur
locks with near-black roots and grizzled silver tips are layered in three irregular tiers on a fur
collar, swept back and outward. It is fullest and longest at the back, where three dull rusted iron
spikes rise through the fur like raised hackles. The ruff is parted at the throat and fastened with a
worn black leather strap looped round a carved bone fang toggle; a few matted locks hang longer over
the chest. One closed mesh, < 4k triangles.
"""
import math
import os
import sys

import bpy  # noqa: F401  (must precede mathutils)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbxkit as K  # noqa: E402

K.use_rig('R6')

NAME = 'AshenGrizzledMane'
np.random.seed(613)                # Asm draws per-part shading noise from the global generator
rng = np.random.default_rng(613)
X, Y, Z = np.eye(3)
TH0 = 0.42                         # half-angle of the parting at the throat (0 = front, +pi/2 = +X)
CLEAR = 0.03                       # collar clearance round the head
FACE_Y, FACE_Z = -0.4, 2.25        # nothing in front of the face above this height
VIEWS = [('preview', (0.0, -0.05, 2.2), (0.72, -1.0, 0.34), 4.4, 50),
         ('front', (0.0, 0.0, 1.7), (0.0, -1.0, 0.1), 5.6, 50),
         ('back', (0.0, 0.15, 2.05), (-0.55, 1.0, 0.28), 4.6, 50),
         ('closeup', (0.0, -0.7, 2.08), (0.38, -1.0, 0.2), 1.45, 50)]


# ------------------------------------------------------------------------------------ materials
ROOT, DEEP = (0.0035, 0.0034, 0.0038), (0.0012, 0.0012, 0.0014)
BODY_D, BODY_L = (0.010, 0.0098, 0.011), (0.032, 0.031, 0.034)
SILVER, WHITE = (0.17, 0.17, 0.176), (0.42, 0.42, 0.43)


def _fur(nb):
    """Fur locks: 'vfac' runs 0 at the root to 1 at the tip, 'across' 0..1 across each lock, so the
    strands follow every lock whatever its direction."""
    rnd, vf, ac = nb.attr('part_rnd'), nb.attr('vfac'), nb.attr('across')
    cx = nb.node('ShaderNodeCombineXYZ')
    nb.feed(cx.inputs['X'], nb.add(nb.mul(ac, 9.0), nb.mul(rnd, 37.0)))
    nb.feed(cx.inputs['Y'], nb.mul(vf, 1.4))
    nb.feed(cx.inputs['Z'], nb.mul(rnd, 11.0))
    sv = cx.outputs['Vector']
    strand = nb.noise(sv, 1.0, 3, 0.55)
    hair = nb.noise(sv, 2.6, 2, 0.45)
    mott = nb.noise(nb.coord('Object'), 7.0, 3, 0.5)
    open_ = nb.mapr(nb.ao(0.08), 0.25, 0.95)
    tone = nb.add(nb.add(nb.mul(strand, 0.7), nb.mul(rnd, 0.35)), nb.mul(mott, 0.3))
    c = nb.mixc(nb.mapr(tone, 0.45, 1.05), BODY_D, BODY_L)
    c = nb.mixc(nb.mapr(vf, 0.04, 0.5, smooth=True), ROOT, c)
    # grizzled tips: the silver starts at a different point on every strand and lock
    start = nb.add(0.56, nb.mul(rnd, 0.22))
    grz = nb.math('MULTIPLY', nb.sub(nb.add(vf, nb.mul(nb.sub(hair, 0.5), 0.9)), start), 4.0, clamp=True)
    c = nb.mixc(grz, c, nb.mixc(nb.mapr(hair, 0.4, 0.7), SILVER, WHITE))
    guard = nb.mul(nb.mapr(hair, 0.66, 0.74), nb.mapr(vf, 0.2, 0.6))       # odd pale guard hairs
    c = nb.mixc(nb.mul(guard, 0.6), c, SILVER)
    c = nb.mixc(nb.mul(nb.inv(open_), 0.9), c, DEEP)
    rough = nb.sub(nb.mapr(strand, 0.3, 0.7, 0.62, 0.82), nb.mul(grz, 0.12))
    h = nb.add(nb.mul(strand, 0.55), nb.mul(hair, 0.45))
    return nb.finish(c, rough, 0.0, nb.bump(h, 0.55, 0.003))


def _bone(nb):
    """Aged, tea-stained bone with grime packed into the carving."""
    co = nb.coord('Object')
    grain = nb.noise(nb.mapping(co, scale=(1.0, 5.0, 5.0)), 40.0, 4, 0.6, distortion=0.4)
    mott = nb.noise(co, 14.0, 4, 0.55)
    pits = nb.voronoi(co, 140.0, 'F1', 'Distance')
    open_ = nb.mapr(nb.ao(0.02), 0.3, 0.95)
    c = nb.mixc(nb.mapr(nb.add(grain, nb.mul(mott, 0.5)), 0.35, 0.9), (0.30, 0.26, 0.19), (0.44, 0.40, 0.31))
    c = nb.mixc(nb.mul(nb.mapr(mott, 0.5, 0.75), 0.6), c, (0.15, 0.11, 0.065))
    c = nb.mixc(nb.mul(nb.mapr(pits, 0.06, 0.0), 0.5), c, (0.11, 0.085, 0.055))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.92), c, (0.025, 0.018, 0.011))
    rough = nb.add(nb.mapr(grain, 0.3, 0.7, 0.38, 0.55), nb.mul(nb.inv(open_), 0.3))
    h = nb.sub(nb.mul(grain, 0.4), nb.mul(nb.mapr(pits, 0.06, 0.0), 0.3))
    return nb.finish(c, rough, 0.0, nb.bump(h, 0.3, 0.002))


K.RECIPES.update({'fur': _fur, 'bone': _bone})


class Asm(K.Asm):
    """K.Asm plus a per-vertex 'across' attribute (0..1 across each fur lock) for the strand texture."""

    def __init__(self):
        super().__init__()
        self.va = []

    def part(self, group, prim, mat, m=None, smooth=180.0, uvs=1.0, vfac=None, across=None):
        self.va.append(np.zeros(len(prim[0])) if across is None else np.asarray(across, float))
        return super().part(group, prim, mat, m, smooth=smooth, uvs=uvs, vfac=vfac)

    def build(self, name):
        ob = super().build(name)
        a = np.concatenate(self.va)
        assert len(a) == len(ob.data.vertices)
        ob.data.attributes.new('across', 'FLOAT', 'POINT').data.foreach_set('value', a.tolist())
        return ob


# ------------------------------------------------------------------------------ body + limits
K.ensure_mannequin()
K.new_scene()
HEAD = K.mannequin_verts('Head')
HZ = np.linspace(HEAD[:, 2].min(), HEAD[:, 2].max(), 49)
_hr = np.array([np.hypot(*HEAD[np.abs(HEAD[:, 2] - z) <= 0.03][:, :2].T).max() for z in HZ])
HR = np.maximum.reduce([_hr, np.r_[_hr[1:], _hr[-1]], np.r_[_hr[0], _hr[:-1]]])   # conservative envelope
R_IN = float(HR.max()) + CLEAR
_parts = [K.R6_PARTS[p] for p in ('Torso', 'LeftArm', 'RightArm')]
BLO = np.min([np.subtract(c, np.divide(s, 2)) for c, s in _parts], 0)
BHI = np.max([np.add(c, np.divide(s, 2)) for c, s in _parts], 0)
_lim, _att = K.LIMITS['Neck'], np.array(K.ATTACH['Neck'])
LO = _att - (_lim[0], _lim[1], _lim[3]) + 0.012
HI = _att + (_lim[0], _lim[2], _lim[4]) - 0.012


def head_r(z):
    return np.interp(z, HZ, HR)


def body_sd(P):
    """Approximate signed distance from points to the R6 torso/arm block and the head."""
    P = np.atleast_2d(np.asarray(P, float))
    c, h = (BLO + BHI) / 2, (BHI - BLO) / 2
    q = np.abs(P - c) - h
    box = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
    zin = (P[:, 2] > HZ[0]) & (P[:, 2] < HZ[-1])
    head = np.where(zin, np.hypot(P[:, 0], P[:, 1]) - head_r(P[:, 2]), np.inf)
    return np.minimum(box, head)


def push(p, m):
    """Move a point to at least m outside the torso/arm block and the head."""
    p = np.array(p, float)
    c, h = (BLO + BHI) / 2, (BHI - BLO) / 2
    for _ in range(3):
        n = np.clip(p, BLO, BHI)
        d = np.linalg.norm(p - n)
        if d == 0.0:                               # inside: out through the nearest face
            q = np.abs(p - c) - h
            ax = int(np.argmax(q))
            p[ax] = c[ax] + math.copysign(h[ax] + m, p[ax] - c[ax])
        elif d < m:
            p = n + (p - n) / d * m
        if HZ[0] < p[2] < HZ[-1]:
            r, rh = math.hypot(p[0], p[1]), float(head_r(p[2])) + m
            if r < rh:
                p[:2] *= rh / max(r, 1e-6)
    return p


def fits(v, face=True):
    ok = bool(np.all(v >= LO) and np.all(v <= HI))
    if face:
        ok = ok and not np.any((v[:, 1] < FACE_Y) & (v[:, 2] > FACE_Z))
    return ok


def smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def radial(th):
    return np.array([math.sin(th), -math.cos(th), 0.0])


def tangent(th):
    return np.array([math.cos(th), math.sin(th), 0.0])


# --------------------------------------------------------------------------------- the collar
def collar_dims(th):
    """Centre radius, centre height and the radial / vertical half sizes of the collar at angle th."""
    kb, ks = (1 - math.cos(th)) / 2, abs(math.sin(th))
    taper = 0.5 + 0.5 * float(smooth01(min(th - TH0, 2 * math.pi - TH0 - th) / 0.45))
    a = (0.112 + 0.058 * ks - 0.012 * kb) * taper
    b = (0.10 + 0.065 * kb + 0.02 * ks) * taper
    return R_IN + a, 2.0 + 0.03 + b, a, b


def on_collar(th, al, depth=0.0):
    """Point and outward normal on the collar surface; al = 0 outwards, +90 deg up, 180 deg inwards."""
    rc, zc, a, b = collar_dims(th)
    r, n = radial(th), K._unit(radial(th) * math.cos(al) / a + Z * math.sin(al) / b)
    p = np.array([0, 0, zc]) + r * (rc + a * math.cos(al)) + Z * b * math.sin(al)
    return p - n * depth, n


A = Asm()
NC, SC = 26, 7
cth = np.linspace(TH0, 2 * math.pi - TH0, NC)
dims = np.array([collar_dims(t) for t in cth])
cpath = np.array([radial(t) * d[0] + Z * d[1] for t, d in zip(cth, dims)])
col = K.tube(cpath, dims[:, 2], sides=SC, caps=('flat', 'flat'), ry=dims[:, 3], ups=[radial(t) for t in cth])
arc = K.rbxgeo._arc(cpath)
ang = 2 * math.pi * np.arange(SC) / SC
A.part('collar', col, 'fur', smooth=180, uvs=0.7,
       vfac=np.tile(0.08 + 0.25 * (1 + np.cos(ang)) / 2, NC), across=np.repeat(arc / 0.12, SC))


# ------------------------------------------------------------------------------------ fur locks
def env_normals(pts, rad, roll):
    """Outward normal of the fur surface along a lock: the lock lies flat in that surface."""
    T = K.frames(pts)[0]
    tang = np.cross(Z, rad)
    out = []
    for t in T:
        nrm = np.cross(t, tang)
        nrm = K._unit(rad + Z if np.linalg.norm(nrm) < 0.25 else nrm)
        if np.dot(nrm, rad + 0.1 * Z) < 0:
            nrm = -nrm
        out.append(nrm * math.cos(roll) + np.cross(t, nrm) * math.sin(roll))
    return out


ACROSS = {3: [0.5, 0.0, 1.0], 4: [0.5, 0.0, 0.5, 1.0]}
WPROF = ([0, 0.25, 0.5, 0.75, 1], [0.55, 1.0, 0.84, 0.5, 0.0])
TPROF = ([0, 0.25, 0.5, 0.75, 1], [1.0, 0.9, 0.7, 0.4, 0.0])
SKIPPED = []


def lock(group, root, d0, L, w, th, rad, grav=0.3, bend=None, sides=3, n=None, roll=0.0, wave=0.0,
         twist=0.0, face=True, tpow=1.0):
    """One tapered, flattened, curved lock of fur from `root` (buried in the collar) along d0.
    Pushed clear of the body and shortened until it fits the Neck bounds."""
    d0 = K._unit(d0)
    bend = -Z * grav if bend is None else bend
    n = n or (4 if L < 0.28 else 5 if L < 0.55 else 6)
    ts = np.linspace(0, 1, n)
    pw, pt = np.interp(ts, *WPROF) * w, np.interp(ts, *TPROF) * th
    side = K._unit(np.cross(d0, Z)) if abs(d0[2]) < 0.95 else np.cross(d0, X)
    extra, why = 0.0, ''
    for _ in range(12):
        pts = root + np.outer(ts, d0 * L) + np.outer(ts ** 2, bend * L)
        if wave:
            pts += np.outer(np.sin(ts * math.pi * 2.2) * ts, side * wave)
        m = 0.016 + pt + 0.3 * pw + extra
        pts[1:] = [push(p, mm) for p, mm in zip(pts[1:], m[1:])]
        ups = env_normals(pts, rad, roll)
        if twist:
            T = K.frames(pts)[0]
            ups = [u * math.cos(twist * t) + np.cross(tt, u) * math.sin(twist * t) for u, t, tt in zip(ups, ts, T)]
        prim = K.tube(pts, pt, sides=sides, caps=('flat', 'point'), ry=pw, ups=ups)
        v = prim[0][sides:]                      # the root ring is buried in the collar
        if not fits(v, face):
            why = f'fit lo={np.round(v.min(0) - LO, 3)} hi={np.round(HI - v.max(0), 3)}'
            L *= 0.9
            continue
        if body_sd(v).min() < 0.01:
            why = f'body {body_sd(v).min():.3f} at {np.round(v[np.argmin(body_sd(v))], 3)}'
            extra += 0.012
            continue
        break
    else:
        SKIPPED.append((group, np.round(root, 2).tolist(), why))
        return None
    vf = np.repeat(ts ** tpow, [sides] * (n - 1) + [1])
    ac = np.r_[np.tile(ACROSS[sides], n - 1), 0.5]
    A.part(group, prim, 'fur', smooth=70, vfac=vf, across=ac)
    return pts


def thetas(n, back, lo, hi, jit=0.35):
    """n angles in [lo, hi], denser towards the back by `back`, stratified with jitter."""
    g = np.linspace(lo, hi, 721)
    cdf = np.cumsum(1 + back * (1 - np.cos(g)) / 2)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    u = (np.arange(n) + 0.5 + rng.uniform(-jit, jit, n)) / n
    return np.interp(np.clip(u, 0, 1), cdf, g)


def sweep(th):
    """Backwards sweep along the collar (zero at the front and back centre)."""
    return tangent(th) * math.sin(th)


# tier 1: the long drape over the shoulders and down the back
for th in thetas(40, 1.2, TH0 + 0.12, 2 * math.pi - TH0 - 0.12):
    kb, ks = (1 - math.cos(th)) / 2, abs(math.sin(th))
    root, nrm = on_collar(th, math.radians(rng.uniform(-48, -22)), 0.05)
    r = radial(th)
    d0 = r * (1.0 - 0.35 * kb ** 2) - Z * (0.22 + 0.55 * kb ** 2) + sweep(th) * 0.45 \
        + tangent(th) * rng.uniform(-0.25, 0.25)
    L = (0.36 + 0.5 * kb ** 1.5) * rng.uniform(0.8, 1.25)
    lock('drape', root, d0, L, rng.uniform(0.05, 0.075), rng.uniform(0.012, 0.018), r,
         grav=rng.uniform(0.35, 0.7), roll=rng.uniform(-0.35, 0.35), sides=3 if L < 0.6 else 4)

# tier 2: the ruff proper, standing out from the collar and swept back
for th in thetas(38, 1.0, TH0 + 0.1, 2 * math.pi - TH0 - 0.1):
    kb, ks = (1 - math.cos(th)) / 2, abs(math.sin(th))
    root, nrm = on_collar(th, math.radians(rng.uniform(-5, 38)), 0.05)
    r = radial(th)
    d0 = r + Z * rng.uniform(0.0, 0.35) + sweep(th) * 0.55 + tangent(th) * rng.uniform(-0.3, 0.3)
    L = (0.3 + 0.2 * kb + 0.08 * ks) * rng.uniform(0.75, 1.3)
    lock('ruff', root, d0, L, rng.uniform(0.045, 0.07), rng.uniform(0.011, 0.016), r,
         grav=rng.uniform(0.15, 0.45), roll=rng.uniform(-0.4, 0.4))

# tier 3: the crest rising round the head, tallest behind it, flaring out at the tips
for th in thetas(34, 1.6, TH0 + 0.16, 2 * math.pi - TH0 - 0.16):
    kb, ks = (1 - math.cos(th)) / 2, abs(math.sin(th))
    kf = max(0.0, math.cos(th)) ** 0.7
    root, nrm = on_collar(th, math.radians(rng.uniform(62, 98)), 0.05)
    r = radial(th)
    d0 = Z * (1.0 - 1.1 * kf) + r * (0.5 - 0.3 * kb + 0.9 * kf) + sweep(th) * 0.35 \
        + tangent(th) * rng.uniform(-0.2, 0.2)
    L = (0.22 + 0.36 * kb + 0.12 * ks) * rng.uniform(0.8, 1.2)
    lock('crest', root, d0, L, rng.uniform(0.04, 0.06), rng.uniform(0.01, 0.014), r,
         bend=r * rng.uniform(0.1, 0.3) - Z * rng.uniform(0.0, 0.15), roll=rng.uniform(-0.4, 0.4))

# matted locks hanging longer over the chest, more on the right than the left
for th, L in ((0.55, 0.62), (0.74, 0.5), (-0.52, 0.72), (-0.66, 0.55), (-0.84, 0.46)):
    th = th % (2 * math.pi)
    root, nrm = on_collar(th, math.radians(-30), 0.04)
    r = radial(th)
    lock('matted', root, r * 0.45 - Z + tangent(th) * rng.uniform(-0.15, 0.15), L, 0.028, 0.02, r,
         grav=0.25, sides=4, n=7, wave=0.03, twist=rng.uniform(1.2, 2.2), roll=0.3, tpow=1.3)

print('SKIPPED locks:', len(SKIPPED))
for sk in SKIPPED:
    print('  ', sk)
if os.environ.get('DEBUGBODY'):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    allv = np.concatenate(A.v)
    labels = [g for g, x in zip(A.part_group, A.v) for _ in range(len(x))]
    for o in K.load_mannequin():
        t = BVHTree.FromObject(o, bpy.context.evaluated_depsgraph_get())
        for i, vv in enumerate(allv):
            loc, nrm, idx, dist = t.find_nearest(Vector(vv))
            if loc is not None and (Vector(vv) - loc).dot(nrm) < 0:
                print('INSIDE', o.name, labels[i], np.round(vv, 3), 'loc', np.round(loc, 3), 'nrm', np.round(nrm, 3))
dev = K.dev_modes(A, NAME, VIEWS)
ob, rep = K.finish_item(A, NAME, 'Neck', 'Neck', 'Ashen Grizzled Mane', meta=dev)
K.previews(NAME, VIEWS, spp=int(os.environ.get('PREVIEW_SPP', 40)))
print('DONE', NAME)
