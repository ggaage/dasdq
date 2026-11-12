"""Ashen Puppeteer's Hand: Shoulder accessory (RightCollarAttachment), fitted to the R6 avatar.

"The wearer is the puppet." A severed bisque ball-jointed-doll hand hovers above the right shoulder,
palm down, gripping a black-lacquered marionette cross control. Taut linen strings drop from the
control's iron screw-eyes onto eyes screwed into the wearer's shoulder; the red thread bound round the
grip runs down as one more string, and one string has snapped and curls loose above its torn stub.
The hand is segmented - palm, phalanges and thumb on visible ball joints - with oxblood-stained
fingertips, black stiletto nails, hairline glaze cracks and a hollow wrist socket trailing frayed
stringing elastic. The lacquer is worn through to its red undercoat on the edges. One closed mesh,
< 4k triangles.
"""
import math
import os
import sys

import bpy  # noqa: F401  (must precede mathutils)
import numpy as np
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbxkit as K  # noqa: E402

K.use_rig('R6')
import rbxsdf as S  # noqa: E402

NAME = 'AshenPuppeteerHand'
np.random.seed(97)                 # Asm draws per-part shading noise from the global generator
rng = np.random.default_rng(97)
X, Y, Z = np.eye(3)
VIEWS = [('preview', (-0.72, -0.05, 2.48), (-0.62, -1.0, 0.36), 4.2, 50),
         ('front', (-0.4, 0.0, 2.15), (0.0, -1.0, 0.1), 5.2, 50),
         ('closeup', (-0.98, -0.02, 2.66), (-0.5, -1.0, 0.62), 1.75, 50)]


# ------------------------------------------------------------------------------------ materials
def _bjd(nb):
    """Glazed bisque doll porcelain: bone white, rosy blush in the joints, sparse hairline cracks and an
    oxblood stain soaked into the fingertips (vfac 0..1)."""
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    vf = nb.attr('vfac')
    n1 = nb.noise(co, 36.0, 4, 0.55)
    mott = nb.noise(co, 7.0, 3, 0.5)
    edge = nb.voronoi(co, 12.0, 'DISTANCE_TO_EDGE', 'Distance')
    crack = nb.mul(nb.mapr(edge, 0.026, 0.004), nb.mapr(nb.noise(co, 5.0, 2, 0.5), 0.5, 0.58))
    open_ = nb.mapr(nb.ao(0.02), 0.35, 0.95)
    c = nb.mixc(nb.mapr(nb.add(n1, nb.mul(rnd, 0.3)), 0.3, 0.75), (0.60, 0.555, 0.49), (0.47, 0.43, 0.37))
    blush = nb.mul(nb.mapr(open_, 0.97, 0.6), nb.mapr(mott, 0.3, 0.7, 0.6, 1.0))
    c = nb.mixc(nb.mul(blush, 0.5), c, (0.42, 0.2, 0.18))
    c = nb.mixc(nb.mul(nb.mapr(open_, 0.6, 0.2), 0.85), c, (0.035, 0.022, 0.02))
    stain = nb.mapr(nb.add(vf, nb.mul(nb.sub(nb.noise(co, 60.0, 3, 0.6), 0.5), 0.55)), 0.36, 0.56)
    core = nb.mapr(nb.add(vf, nb.mul(nb.sub(n1, 0.5), 0.4)), 0.7, 0.95)
    c = nb.mixc(stain, c, (0.16, 0.018, 0.018))
    c = nb.mixc(nb.mul(core, 0.8), c, (0.06, 0.005, 0.007))
    c = nb.mixc(nb.mul(crack, 0.9), c, (0.05, 0.035, 0.03))
    rough = nb.add(nb.mapr(n1, 0.3, 0.7, 0.24, 0.38), nb.add(nb.mul(crack, 0.35), nb.mul(stain, 0.12)))
    h = nb.sub(nb.mul(n1, 0.15), nb.mul(crack, 0.8))
    return nb.finish(c, rough, 0.0, nb.bump(h, 0.35, 0.002))


def _lacquer(nb):
    """Black urushi lacquer over a red undercoat, worn through on edges and scuffs, bare wood at the worst."""
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    grain = nb.noise(co, 45.0, 5, 0.6, distortion=2.5)
    scuff = nb.noise(co, 18.0, 4, 0.65)
    scratch = nb.noise(nb.mapping(co, scale=(1.0, 1.0, 9.0)), 90.0, 2, 0.5)
    open_ = nb.mapr(nb.ao(0.03), 0.35, 0.95)
    edge_m = nb.mul(K.edge_mask(nb, 0.005, 0.03, 0.22), open_)
    worn = nb.add(nb.mul(edge_m, 0.9), nb.mul(nb.sub(scuff, 0.5), 0.9))
    wear = nb.mapr(worn, 0.3, 0.6)
    bare = nb.mapr(worn, 0.62, 0.85)
    c = nb.mixc(nb.mapr(nb.add(grain, nb.mul(rnd, 0.2)), 0.35, 0.7), (0.0045, 0.004, 0.004), (0.014, 0.010, 0.009))
    c = nb.mixc(nb.mul(wear, 0.95), c, (0.15, 0.016, 0.014))
    c = nb.mixc(nb.mul(bare, 0.85), c, (0.07, 0.04, 0.022))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.7), c, (0.002, 0.002, 0.002))
    rough = nb.add(nb.mapr(scratch, 0.35, 0.75, 0.1, 0.24), nb.mul(wear, 0.4))
    h = nb.sub(nb.mul(nb.add(nb.mul(grain, 0.5), nb.mul(scratch, 0.5)), 0.3), nb.mul(wear, 0.5))
    return nb.finish(c, rough, 0.0, nb.bump(h, 0.2, 0.003))


def _red_thread(nb):
    co = nb.coord('Object')
    fib = nb.noise(co, 180.0, 3, 0.6)
    mott = nb.noise(co, 30.0, 3, 0.5)
    open_ = nb.mapr(nb.ao(0.015), 0.35, 0.95)
    c = nb.mixc(nb.mapr(nb.add(nb.mul(fib, 0.6), nb.mul(mott, 0.4)), 0.3, 0.75), (0.07, 0.004, 0.006),
                (0.2, 0.016, 0.016))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.7), c, (0.02, 0.002, 0.002))
    return nb.finish(c, nb.mapr(fib, 0.3, 0.7, 0.65, 0.85), 0.0, nb.bump(fib, 0.3, 0.002))


def _elastic(nb):
    """Aged off-white braided doll-stringing elastic."""
    co = nb.coord('Object')
    braid = nb.noise(co, 140.0, 2, 0.5)
    mott = nb.noise(co, 25.0, 3, 0.5)
    open_ = nb.mapr(nb.ao(0.02), 0.35, 0.95)
    c = nb.mixc(nb.mapr(nb.add(nb.mul(braid, 0.5), nb.mul(mott, 0.5)), 0.3, 0.75), (0.16, 0.14, 0.11),
                (0.36, 0.33, 0.27))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.75), c, (0.02, 0.016, 0.012))
    return nb.finish(c, 0.85, 0.0, nb.bump(braid, 0.4, 0.002))


K.RECIPES.update({'bjd': _bjd, 'lacquer': _lacquer, 'red_thread': _red_thread, 'elastic': _elastic})

# ------------------------------------------------------------------------------------- layout
# Control frame: u along the crossbars (the way the fingers point), v along the main bar, w up; origin on
# the main-bar axis under the palm. Hand frame: x from the wrist towards the knuckles, y towards the
# little finger, z out of the back of the hand. It is a left hand, so the thumb sits on the -v side.
RB = 0.033                                     # main-bar radius at the grip
GRIP = np.array([-1.0, 0.02, 2.78])            # world position of the grip
YAW, TILT_U, TILT_V = math.radians(-40), math.radians(5), math.radians(-6)
BETA = math.radians(16)                        # palm pitched down towards the fingers
XC, ZC = 0.205, -(0.031 + RB + 0.003)          # bar axis in the hand frame: under the palm, behind the knuckles
RC = np.array(Matrix.Rotation(YAW, 3, 'Z') @ Matrix.Rotation(TILT_U, 3, 'X') @ Matrix.Rotation(TILT_V, 3, 'Y'))
MC = np.eye(4)
MC[:3, :3], MC[:3, 3] = RC, GRIP
U, V, W = RC.T                                 # world directions of the control axes
xh = np.array([math.cos(BETA), 0.0, -math.sin(BETA)])
yh = Y.copy()
zh = np.cross(xh, yh)
MH = np.eye(4)
MH[:3, 0], MH[:3, 1], MH[:3, 2], MH[:3, 3] = xh, yh, zh, -(XC * xh + ZC * zh)
MW = MC @ MH                                   # hand frame -> world

# fingers: knuckle (hand frame), phalanx lengths, radii at knuckle/PIP/DIP/tip, extra clearance per phalanx
FINGERS = [('index', (0.262, -0.064, 0.002), (0.098, 0.064, 0.052), (0.0215, 0.0195, 0.0175, 0.015), (0.004, 0.008, 0.014)),
           ('middle', (0.274, -0.021, 0.004), (0.106, 0.070, 0.056), (0.022, 0.020, 0.018, 0.0155), (0.003, 0.003, 0.005)),
           ('ring', (0.265, 0.022, 0.002), (0.100, 0.066, 0.053), (0.0212, 0.0193, 0.0173, 0.015), (0.003, 0.004, 0.006)),
           ('pinky', (0.243, 0.061, -0.003), (0.080, 0.052, 0.045), (0.019, 0.0172, 0.0155, 0.0135), (0.008, 0.024, 0.04))]
THUMB = ((0.13, -0.1, -0.032), (0.07, 0.058), (0.0225, 0.0205, 0.0165), (0.004, 0.007))
SOCKET, SOCKET_R = np.array([-0.07, 0.0, 0.003]), 0.033


def cw(p):
    """Control frame -> world."""
    return K.xf(MC, p)


K.ensure_mannequin()
K.new_scene()
A = K.Asm()


# ----------------------------------------------------------------------------------------- hand
def palm_sdf(P):
    d = None
    for (_, kn, _, rad, _), y0 in zip(FINGERS, (-0.042, -0.014, 0.014, 0.04)):
        s = S.capsule(P, (0.04, y0, 0.002), np.asarray(kn) - (0.024, 0.0, 0.0), rad[0] + 0.006, rad[0] + 0.008)
        d = s if d is None else S.smin(d, s, 0.03)
    tb = np.asarray(THUMB[0])
    d = S.smin(d, S.capsule(P, (0.05, -0.04, -0.012), tb - (0.012, -0.008, -0.004), 0.03, 0.024), 0.03)  # thenar
    d = S.smin(d, S.ellipsoid(P, (0.105, 0.05, -0.012), (0.085, 0.03, 0.026)), 0.025)                   # hypothenar
    Q = P * np.array([1.0, 0.83, 1.0])
    stump = S.smax(S.capsule(Q, (0.07, 0.0, 0.002), (-0.05, 0.0, 0.004), 0.045), -(P[..., 0] + 0.058), 0.006)
    d = S.smin(d, stump, 0.035)
    return S.cut(d, S.sphere(P, SOCKET, SOCKET_R), 0.004)          # the hollow ball socket at the wrist


palm = S.mesh(palm_sdf, (-0.075, -0.14, -0.085), (0.315, 0.105, 0.055), 0.005, 380, name='palm')
dsock = np.linalg.norm(palm[0] - SOCKET, axis=1)
A.part('palm', palm, 'bjd', MW, vfac=np.clip((SOCKET_R + 0.007 - dsock) / 0.01, 0.0, 1.0) * 0.7)


def wrap(j0, segs, sign=1.0):
    """Joint chain in the control's (u, w) plane hugging the bar (axis at the origin): each joint lands where
    its phalanx's midpoint clears the bar by the phalanx radius plus a gap. sign -1 wraps the other way."""
    pts = [np.asarray(j0, float)]
    for c, r, gap in segs:
        R = math.hypot(RB + r + gap, c / 2)
        J = pts[-1]
        d = float(np.linalg.norm(J))
        e = J / d
        a = (R * R - c * c + d * d) / (2 * d)
        if a > R:
            print(f'WARN wrap: joint {len(pts)} cannot reach the bar ({a:.3f} > {R:.3f})')
        h = math.sqrt(max(R * R - a * a, 0.0))
        pts.append(e * min(a, R) + sign * np.array([e[1], -e[0]]) * h)
    return pts


def dorsal(p):
    """World direction away from the main-bar axis (the back of a finger wrapped round it)."""
    q = p - GRIP
    return K._unit(q - V * np.dot(q, V))


def phalanx(group, a, b, r0, r1, ga, gb, stain=False):
    """Tapered phalanx with domed ends from joint a to joint b (world), stopping short of the joint balls."""
    d = K._unit(b - a)
    L = float(np.linalg.norm(b - a)) - ga - gb
    prof = [(0.0, L), (0.68 * r1, L - 0.22 * r1), (r1, L - 0.75 * r1), (r0, 0.75 * r0), (0.68 * r0, 0.22 * r0),
            (0.0, 0.0)]
    prim = K.lathe(prof, 6)
    vf = np.clip((prim[0][:, 2] / L - 0.15) / 0.6, 0.0, 1.0) if stain else None
    dors = dorsal((a + b) / 2)
    A.part(group, prim, 'bjd', K.mat_trs(a + d * ga, (tuple(d), tuple(dors)), (1.0, 0.9, 1.0)), vfac=vf)
    return a + d * ga, d, L, dors


def nail(group, base, d, L, dors, r0, r1):
    """Glossy black stiletto nail on the back of a fingertip, running out past the tip."""
    pts = [base + d * 0.4 * L + dors * 0.8 * r0, base + d * 0.8 * L + dors * 0.74 * r1,
           base + d * (L + 0.02) + dors * 0.12 * r1]
    A.part(group, K.tube(pts, [0.0032, 0.0034, 0.0], ry=[0.6 * r0, 0.66 * r1, 0.0], sides=6, caps=('flat', 'point'),
                         up=tuple(dors)), 'gem', smooth=60)


def digit(group, knuckle_ctrl, lengths, rads, gaps, sign=1.0, knuckle_ball=1.0):
    """Ball-jointed digit wrapped round the bar: phalanges, joint balls, stained tip and nail."""
    v0 = knuckle_ctrl[1]
    mids = [(rads[i] + rads[i + 1]) / 2 for i in range(len(lengths))]
    j2 = wrap((knuckle_ctrl[0], knuckle_ctrl[2]), list(zip(lengths, mids, gaps)), sign)
    joints = [cw((p[0], v0, p[1])) for p in j2]
    n = len(lengths)
    balls = [0.8 * rads[i] * (knuckle_ball if i == 0 else 1.0) for i in range(n)]
    for i in range(n):
        A.part(group, K.sphere(balls[i], 6, 4 if i == 0 else 3), 'bjd', K.mat_trs(joints[i]))
        ga = 0.5 * balls[i]
        gb = 0.5 * balls[i + 1] if i + 1 < n else 0.0
        last = i == n - 1
        base, d, L, dors = phalanx(group, joints[i], joints[i + 1], rads[i], rads[i + 1], ga, gb, stain=last)
        if last:
            nail(group, base, d, L, dors, rads[i], rads[i + 1])
    return joints


for name, kn, lens, rads, gaps in FINGERS:
    digit('fingers', K.xf(MH, kn), lens, rads, gaps, knuckle_ball=1.12)
tb, tl, tr, tg = THUMB
thumb_ctrl = K.xf(MH, tb) - np.array([0.0, 0.015, 0.0])
digit('thumb', thumb_ctrl, tl, tr, tg, sign=-1.0, knuckle_ball=1.05)

# frayed stringing elastic trailing out of the wrist socket
out = K._unit(-(MW[:3, 0]))
side = K._unit(np.cross(Z, out))
c0 = K.xf(MW, (-0.028, 0.0, 0.003))
c1 = K.xf(MW, (-0.082, 0.004, 0.0))
c2 = c1 + out * 0.035 - Z * 0.028
c3 = c2 + out * 0.02 - Z * 0.07 + side * 0.016
c4 = c3 + out * 0.006 - Z * 0.075 - side * 0.012
cord = K.spline([c0, c1, c2, c3, c4], 4)
A.part('cord', K.tube(cord, 0.0095, sides=5), 'elastic')
for k in range(4):
    a = 2 * math.pi * k / 4 + rng.uniform(-0.4, 0.4)
    sd = K._unit(-Z + 0.55 * (math.cos(a) * out + math.sin(a) * side))
    ln = rng.uniform(0.03, 0.05)
    p0 = c4 + (math.cos(a) * out + math.sin(a) * side) * 0.004
    A.part('cord', K.tube([p0 + Z * 0.004, p0 + sd * ln * 0.5, p0 + sd * ln + side * 0.005], [0.0036, 0.0028, 0.0],
                          sides=3, caps=('flat', 'point')), 'elastic')


# -------------------------------------------------------------------------------------- control
def bar_r(prof, t):
    return float(np.interp(t, prof[0], prof[1]))


MAIN = ([-0.42, -0.34, -0.15, 0.15, 0.34, 0.42], [0.028, 0.036, RB, RB, 0.036, 0.028])
CROSS = (-0.3, 0.25, 0.022)                    # front crossbar: v, half length, droop at the ends
TAIL = (0.32, 0.17, 0.012)
CROSS_R = ([-1.0, -0.65, -0.2, 0.2, 0.65, 1.0], [0.022, 0.027, 0.031, 0.031, 0.027, 0.022])


def cross_pt(bar, u):
    v, half, droop = bar
    return np.array([u, v, -droop * (u / half) ** 2])


A.part('control', K.tube([cw((0, v, 0)) for v in MAIN[0]], np.array(MAIN[1]), sides=8, phase=math.pi / 8,
                         up=tuple(W)), 'lacquer', smooth=60)
EYES = []
for bar in (CROSS, TAIL):
    v, half, droop = bar
    s = half / 0.25
    ts = np.array(CROSS_R[0])
    rads = np.array(CROSS_R[1]) * (1.0 if bar is CROSS else 0.86)
    A.part('control', K.tube([cw(cross_pt(bar, t * half)) for t in ts], rads, sides=8, phase=math.pi / 8,
                             up=tuple(W)), 'lacquer', smooth=60)
    A.part('control', K.sphere(0.013, 6, 3), 'chain_iron', K.mat_trs(cw((0, v, 0.035)), None, (1.0, 1.0, 0.55)))
    for sgn in (-1, 1):
        end = cw(cross_pt(bar, sgn * half))
        tang = K._unit(cw(cross_pt(bar, sgn * half)) - cw(cross_pt(bar, sgn * (half - 0.02))))
        re = rads[-1]
        fer = [(0.0, 0.5 * re), (0.75 * re, 0.38 * re), (1.15 * re, 0.0), (1.15 * re, -1.1 * re), (1.06 * re, -1.22 * re)]
        A.part('control', K.lathe(fer, 7), 'chain_iron', K.mat_trs(end, (tuple(tang), tuple(W))), smooth=50)
        ue = sgn * (half - 0.032)
        bot = cw(cross_pt(bar, ue)) - W * bar_r((ts * half, rads), ue) * 0.92
        re_eye, we = 0.0125, 0.0042
        ec = bot - W * (0.007 + re_eye)
        A.part('control', K.tube([bot + W * 0.008, ec + W * re_eye * 0.8], 0.0038, sides=4), 'chain_iron')
        A.part('control', K.ring(re_eye, re_eye, we, segs=8, sides=3), 'chain_iron', K.frame(ec, W, V))
        EYES.append(ec - W * (re_eye - 0.002))
for sgn in (-1, 1):
    end = cw((0, sgn * 0.42, 0))
    re = MAIN[1][-1]
    fer = [(0.0, 0.5 * re), (0.75 * re, 0.38 * re), (1.15 * re, 0.0), (1.15 * re, -1.1 * re), (1.06 * re, -1.22 * re)]
    A.part('control', K.lathe(fer, 7), 'chain_iron', K.mat_trs(end, (tuple(V * sgn), tuple(W))), smooth=50)

# red thread bound round the grip beside the little finger; its tail runs down as the red string
WRAP_V, TURNS, PER = (0.098, 0.168), 4, 8
hel = []
for i in range(TURNS * PER + 5):
    t = i / (TURNS * PER)
    ph = 2 * math.pi * TURNS * min(t, 1.0) + (2 * math.pi * (t - 1.0) * TURNS if t > 1 else 0.0)
    vv = WRAP_V[0] + (WRAP_V[1] - WRAP_V[0]) * min(t, 1.0)
    rr = bar_r(MAIN, vv) + 0.0048
    hel.append(cw((rr * math.sin(ph), vv, rr * math.cos(ph))))
hel = np.array(hel)


# ------------------------------------------------------------------------- shoulder anchors + strings
def anchor_xy(p, lo=(-0.94, -0.43), hi=(-0.68, 0.43)):
    return np.array([min(max(p[0], lo[0]), hi[0]), min(max(p[1], lo[1]), hi[1]), 2.0])


TOP = 2.0
RO_Z = TOP + 0.012                             # rosette underside: clear of the torso


def screw_eye(base_xy, toward):
    """Iron screw-eye on a rosette on the shoulder top; returns the point a string ties to."""
    b = np.array([base_xy[0], base_xy[1], RO_Z])
    A.part('anchors', K.lathe([(0.0, 0.012), (0.017, 0.010), (0.022, 0.005), (0.022, 0.0)], 6), 'chain_iron',
           K.mat_trs(b), smooth=50)
    hd = K._unit(np.array([toward[0], toward[1], 0.0]) + 1e-6)
    re, we = 0.0135, 0.0042
    ec = b + Z * (0.012 + re - 0.003)
    A.part('anchors', K.ring(re, re, we, segs=8, sides=3), 'chain_iron', K.frame(ec, Z, np.cross(Z, hd)))
    return ec + Z * (re - 0.003)


SNAP = 3                                       # the tail bar's outer eye: its string has snapped
strings = []
for i, e in enumerate(EYES):
    a = anchor_xy(e)
    knot = screw_eye(a, e - a)
    if i == SNAP:
        snap_anchor = knot
        continue
    strings.append((e, knot, 'string'))
red_a = anchor_xy(hel[-1])
red_knot = screw_eye(red_a, hel[-1] - red_a)
A.part('strings', K.tube(np.vstack([hel, red_knot]), 0.0052, sides=3), 'red_thread', smooth=70)
for top, bot, mat in strings:
    A.part('strings', K.tube([top, bot], 0.0055, sides=4), mat)
    for p in (top, bot):
        A.part('strings', K.sphere(0.0085, 6, 3), mat, K.mat_trs(p))

# the snapped string: hangs from its eye and curls loose; a torn stub stays on the shoulder eye
e = EYES[SNAP]
drift = K._unit(np.array([-U[0], -U[1], 0.0]))
ax = K._unit(-Z + drift * 0.3)
e1 = K._unit(np.cross(ax, Z))
e2 = np.cross(ax, e1)
ctrl = [e, e - Z * 0.07 + drift * 0.012, e - Z * 0.15 + drift * 0.03]
c0 = ctrl[-1] - Z * 0.03
curl = [c0 + (0.028 - 0.008 * t) * (math.cos(a) * e1 + math.sin(a) * e2) + ax * 0.045 * t
        for t, a in zip(np.linspace(0, 1, 14), np.linspace(0.0, 2.6 * math.pi, 14))]
loose = np.vstack([K.spline(ctrl + [curl[0]], 3), curl[1:]])
A.part('strings', K.tube(loose, 0.0052, sides=3), 'string', smooth=70)
A.part('strings', K.sphere(0.0085, 6, 3), 'string', K.mat_trs(e))
tip = loose[-1]
for k in range(3):
    sd = K._unit(ax + 0.7 * (math.cos(2.1 * k) * e1 + math.sin(2.1 * k) * e2))
    A.part('strings', K.tube([tip, tip + sd * 0.012, tip + sd * 0.024 - Z * 0.004], [0.0028, 0.002, 0.0], sides=3,
                             caps=('flat', 'point')), 'string')
sd = K._unit(Z + drift * 0.8)
stub = [snap_anchor, snap_anchor + Z * 0.03, snap_anchor + Z * 0.045 + sd * 0.02, snap_anchor + Z * 0.035 + sd * 0.045]
A.part('strings', K.tube(K.spline(stub, 3), 0.0052, sides=3), 'string', smooth=70)
A.part('strings', K.sphere(0.0085, 6, 3), 'string', K.mat_trs(snap_anchor))
for k in range(2):
    tp = stub[-1]
    sd2 = K._unit(sd - Z * 0.8 + (k - 0.5) * 0.9 * np.cross(Z, sd))
    A.part('strings', K.tube([tp, tp + sd2 * 0.016], [0.0026, 0.0], sides=3, caps=('flat', 'point')), 'string')

print('GRIP', GRIP, 'EYES', np.round(EYES, 3).tolist())
body = K.dev_modes(A, NAME, VIEWS)
ob, rep = K.finish_item(A, NAME, 'Shoulder', 'RightCollar', "Ashen Puppeteer's Hand", meta=body)
K.previews(NAME, VIEWS, spp=int(os.environ.get('PREVIEW_SPP', 40)))
print('DONE', NAME)
