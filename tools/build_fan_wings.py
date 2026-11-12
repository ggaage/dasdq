"""Ashen Lace Fan Wings: Back accessory (BodyBackAttachment), fitted to the default R6 avatar.

Two antique folding fans opened behind the shoulder blades like a pair of wings. Each leaf is real
pleated geometry - black silk with a damask print and an oxblood lace border cut to a pointed-scallop
edge - over black-lacquered sticks that splay like ribs below it. The guard sticks are wider, pierced
with a gothic lancet and run out past the lace into claw points; each pivot is a tarnished silver rivet
boss ringed with aged pearls. The pair is placed by hand, not mirrored: the left fan is opened wider,
swept further back and has a panel of its leaf torn away; from the right fan's rivet alone hangs one
short chain with a small tarnished heart. One closed mesh, < 4k triangles.
"""
import math
import os
import sys

import bpy  # noqa: F401  (must precede mathutils)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rbxkit as K  # noqa: E402
import rbxsdf as S  # noqa: E402

K.use_rig('R6')

NAME = 'AshenLaceFanWings'
np.random.seed(613)                # Asm draws per-part shading noise from the global generator
rng = np.random.default_rng(613)
X, Y, Z = np.eye(3)
T, GT = 0.016, 0.026               # leaf and guard-stick thickness
SH = 0.0085                        # inner stick half thickness
LA, LB, LR = 0.03, 0.019, 0.0062   # chain link centre-line half length / half width, wire radius
PITCH = 2 * (LA - LR)
OXB = (0.15, 0.011, 0.015)         # oxblood lace thread (linear)
OXB_D = (0.055, 0.005, 0.007)
MED = [(0.0, 0.062), (0.026, 0.056), (0.046, 0.04), (0.054, 0.024), (0.072, 0.019), (0.128, 0.014),
       (0.138, 0.004), (0.12, -0.004), (0.0, -0.006)]   # rivet boss + pearl plate, (r, z) top -> bottom
RP, NPEARL, RPR = 0.022, 10, 0.1   # pearl radius, count, ring radius


def rotm(axis, ang):
    a = K._unit(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * k @ k


def fdir(side, phi):
    """Stick direction in the fan's design plane (XZ): phi from +Z towards the fan's own side."""
    return side * math.sin(phi) * X + math.cos(phi) * Z


def ang(a, b):
    return math.atan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))


class Fan:
    """A folding fan: pivot, stick angles, leaf outline and pleat depth, oriented by a tilt about the
    axis across its bisector (sweeps the leaf back from the body) and a twist about the bisector."""

    def __init__(self, key, side, pivot, phi0, phi1, sticks, tilt, twist, radii, r0, alpha0, scallop,
                 tear=None):
        self.key, self.side, self.P, self.n = key, side, np.asarray(pivot, float), sticks
        self.phi0, self.phi1 = math.radians(phi0), math.radians(phi1)
        self.dphi = (self.phi1 - self.phi0) / (sticks - 1)
        self.al = self.dphi / 2
        self.a0 = math.radians(alpha0)            # half-pleat angle of the leaf laid flat
        cb = math.tan(self.al) / math.tan(self.a0)
        sb = math.sqrt(max(0.0, 1.0 - cb * cb))
        # fold height per stud of radius: each half pleat is a flat sector hinged on its stick line
        self.kd = math.sin(self.a0) * sb / math.sqrt(math.cos(self.a0) ** 2 + (math.sin(self.a0) * cb) ** 2)
        self.radii, self.r0, self.sc, self.tear = radii, r0, scallop, tear
        self.J = 2 * (sticks - 1)
        b = fdir(side, (self.phi0 + self.phi1) / 2)
        rt = rotm(np.cross(b, Y), math.radians(tilt))
        self.M = rotm(rt @ b, math.radians(twist)) @ rt
        self.N, self.E1, self.E2 = self.M @ Y, self.M @ Z, self.M @ (side * X)
        if tear:
            self.jag = rng.uniform(-1.0, 1.0, 13)

    def world(self, r, phi, w):
        return self.P + self.M @ (r * fdir(self.side, phi) + w * Y)

    def r_base(self, phi):
        r0, rm, r1, tp = self.radii
        t = (phi - self.phi0) / (self.phi1 - self.phi0)
        if t <= tp:
            return r0 + (rm - r0) * math.sin(0.5 * math.pi * t / tp)
        return r1 + (rm - r1) * math.cos(0.5 * math.pi * (t - tp) / (1 - tp))

    def scallop(self, phi):
        k = (phi - self.phi0) / self.dphi
        x = k - math.floor(k)
        return self.sc * max(0.0, 1 - abs(2 * x - 1)) ** 0.55

    def r_edge(self, phi):
        return self.r_base(phi) + self.scallop(phi)

    def tear_depth(self, phi):
        if not self.tear:
            return 0.0
        a, b, depth = self.tear
        a, b = self.phi0 + a * self.dphi, self.phi0 + b * self.dphi
        if not a < phi < b:
            return 0.0
        t = (phi - a) / (b - a)
        env = math.sin(math.pi * t)
        jag = np.interp(t, np.linspace(0, 1, len(self.jag)), self.jag)
        return depth * env ** 0.55 * (0.82 + 0.18 * math.sin(7.0 * t)) + 0.07 * jag * env ** 0.3

    def torn(self, j):
        if not self.tear:
            return False
        a, b, _ = self.tear
        return a * 2 < j + 1 and j < b * 2

    def h(self, r, phi):
        """Leaf mid-surface height above the fan plane: 0 on stick lines, kd*r on folds."""
        k = (phi - self.phi0) / self.al
        j = min(max(int(math.floor(k + 1e-9)), 0), self.J - 1)
        dl = (k - j) * self.al
        s = math.sin(dl) if j % 2 == 0 else math.sin(self.al - dl)
        return self.kd * r * s / math.sin(self.al)

    def line_dir(self, j):
        phi = self.phi0 + j * self.al
        return K._unit(fdir(self.side, phi) + (self.kd * Y if j % 2 else 0.0))

    # ------------------------------------------------------------------------------- geometry
    def leaf(self, A, m=3, mt=7):
        phis, bidx, panel = [], [], []
        for j in range(self.J):
            mm = mt if self.torn(j) else m
            bidx.append(len(phis))
            for k in range(mm):
                phis.append(self.phi0 + self.al * (j + k / mm))
                panel.append(j)
        bidx.append(len(phis))
        phis.append(self.phi1)
        panel.append(self.J - 1)
        no = len(phis)
        lines = [self.phi0 + self.al * j for j in range(self.J + 1)]
        r_int = np.array([self.r_edge(p) for p in phis])
        r_out = r_int - np.array([self.tear_depth(p) for p in phis])
        R = np.r_[r_out, np.full(self.J + 1, self.r0)]
        PH = np.r_[phis, lines]
        H = np.array([self.h(r, p) for r, p in zip(R, PH)])
        front = np.array([self.world(r, p, hh + T / 2) for r, p, hh in zip(R, PH, H)])
        back = np.array([self.world(r, p, hh - T / 2) for r, p, hh in zip(R, PH, H)])
        tris = []
        for j in range(self.J):
            ia, ib = no + j, no + j + 1
            o = list(range(bidx[j], bidx[j + 1] + 1))
            half = (len(o) - 1) // 2
            tris += [[ia, o[k], o[k + 1]] for k in range(half)] + [[ia, o[half], ib]]
            tris += [[ib, o[k], o[k + 1]] for k in range(half, len(o) - 1)]
        for t in tris:
            if np.dot(np.cross(front[t[1]] - front[t[0]], front[t[2]] - front[t[0]]), self.N) < 0:
                t[1], t[2] = t[2], t[1]
        # UVs: the leaf unfolded flat (true pleat widths), studs
        L = [self.line_dir(j) for j in range(self.J + 1)]
        cum = np.r_[0.0, np.cumsum([ang(L[j], L[j + 1]) for j in range(self.J)])]
        uv = []
        for i, (r, p, hh) in enumerate(zip(R, PH, H)):
            q = r * fdir(self.side, p) + hh * Y
            j = panel[i] if i < no else i - no
            th = cum[j] + (ang(q, L[j]) if i < no else 0.0)
            uv.append((np.linalg.norm(q) * math.cos(th), np.linalg.norm(q) * math.sin(th)))
        vf = np.r_[r_int - r_out, [self.r_edge(p) - self.r0 for p in lines]]
        A.part('leaf_' + self.key, K.solid_from_surface(front, tris, back, np.array(uv)), 'silk_' + self.key,
               smooth=30, vfac=np.r_[vf, vf])

    def sticks(self, A):
        wc = -T / 2 - SH + 0.0025
        for i in range(1, self.n - 1):
            ph = self.phi0 + i * self.dphi
            wid = 0.3 * self.r0 * self.dphi
            rs = [-0.05, self.r0 - 0.06, self.r0 + 0.035, self.r_base(ph) - 0.06]
            ws = [wc - 0.0011 * i, wc, wc, wc]
            pts = [self.world(r, ph, w) for r, w in zip(rs, ws)]
            A.part('sticks', K.tube(pts, [SH, SH, SH * 0.8, SH * 0.6], sides=4, caps=('point', 'point'), up=self.N,
                                    ry=[0.01, wid, 0.014, 0.008]), 'lacquer', smooth=40)

    def guards(self, A):
        for end in (0, 1):
            j, ph = (0, self.phi0) if end == 0 else (self.J - 1, self.phi1)
            ls, lf = self.line_dir(j if end == 0 else j + 1), self.line_dir(j + 1 if end == 0 else j)
            nrm = K._unit(np.cross(ls, lf))
            nrm = nrm if np.dot(nrm, Y) > 0 else -nrm
            ey = K._unit(lf - ls * np.dot(lf, ls))                 # across the guard, into the leaf
            o = self.P + self.M @ nrm * (T / 2 + GT / 2 - 0.0015)
            outline, holes = guard_outline(self.r0, self.r_base(ph))
            A.part('guards', K.prism(outline, GT, holes), 'lacquer', K.axes(o, self.M @ ls, self.M @ ey), smooth=35)

    def pivot(self, A):
        base = self.P + self.N * (T / 2 + GT - 0.004)
        m = K.mat_trs(base, (tuple(self.N), tuple(self.E1)))
        A.part('pivot', K.lathe(MED, segs=14), 'silver', m, smooth=50)
        for k in range(NPEARL):
            a = 2 * math.pi * (k + 0.5) / NPEARL
            c = (RPR * math.cos(a), RPR * math.sin(a), 0.0155 + RP * 0.8)
            A.part('pearls', K.sphere(RP, 6, 3), 'pearl', m @ K.mat_trs(c), smooth=180)
        return base


def guard_outline(r0, re):
    """Guard stick in its own plane: x from the pivot outwards, y across (+ towards the leaf). Wide pierced
    gorge, slim over the leaf, then a claw point curling away from the leaf past the lace edge."""
    xs = [-0.075, -0.03, 0.09, 0.2, 0.42, r0 - 0.05, r0 + 0.07, re - 0.12, re, re + 0.09, re + 0.21]
    hw = [0.03, 0.043, 0.03, 0.046, 0.05, 0.04, 0.026, 0.022, 0.019, 0.011, 0.0]
    xc = re - 0.12

    def yc(x):
        return -0.075 * max(0.0, (x - xc) / (xs[-1] - xc)) ** 2

    up = [(x, yc(x) + w) for x, w in zip(xs, hw)]
    lo = [(x, yc(x) - w) for x, w in zip(xs[:-1], hw[:-1])][::-1]
    wl, x0, x1 = 0.017, 0.2, 0.47
    lancet = [(x0 + 0.015, -wl), (x1 - 0.06, -wl), (x1 - 0.024, -0.0105), (x1, 0.0), (x1 - 0.024, 0.0105),
              (x1 - 0.06, wl), (x0 + 0.015, wl), (x0, 0.0)]
    return up + lo, [lancet]


# --------------------------------------------------------------------------------------- materials
def vec(nb, op, a, b):
    n = nb.node('ShaderNodeVectorMath', operation=op)
    nb.feed(n.inputs[0], a)
    nb.feed(n.inputs[1], b)
    return n.outputs['Value'] if op == 'DOT_PRODUCT' else n.outputs['Vector']


def band(nb, v, c, w):
    """1 within about w of c (soft edged)."""
    return nb.mapr(nb.absf(nb.sub(v, c)), w, w * 0.55)


def silk_recipe(fan):
    def build(nb):
        co = nb.coord('Object')
        d = vec(nb, 'SUBTRACT', co, tuple(fan.P))
        a, b = vec(nb, 'DOT_PRODUCT', d, tuple(fan.E1)), vec(nb, 'DOT_PRODUCT', d, tuple(fan.E2))
        r = nb.math('SQRT', nb.add(nb.mul(a, a), nb.mul(b, b)))
        k = nb.mul(nb.sub(nb.math('ARCTAN2', b, a), fan.phi0), 1.0 / fan.dphi)   # pleats from the first stick
        xp = nb.frac(k)
        fold = nb.inv(nb.absf(nb.sub(nb.mul(xp, 2.0), 1.0)))                   # 0 on sticks, 1 on folds
        pw = nb.mul(r, 2 * fan.a0)                                              # pleat width laid flat
        s = nb.mul(nb.sub(xp, 0.5), pw)                                         # across the pleat (studs)
        rho = nb.attr('vfac')                                                   # in from the lace edge
        rb = nb.sub(rho, nb.mul(nb.math('POWER', fold, 0.55), fan.sc))          # in from the scallop roots
        mott = nb.noise(co, 30.0, 4, 0.6)
        fine = nb.noise(co, 160.0, 3, 0.5)
        # ---- damask: one quatrefoil medallion column per pleat, half-drop rows
        v = nb.add(nb.mul(nb.sub(r, fan.r0 + 0.12), 1 / 0.22), nb.mul(nb.math('MODULO', nb.math('FLOOR', k), 2.0), 0.5))
        my = nb.mul(nb.sub(nb.frac(v), 0.5), 0.22)
        dm = nb.math('SQRT', nb.add(nb.mul(s, s), nb.mul(my, my)))
        am = nb.math('ARCTAN2', my, s)
        lobe = nb.mul(nb.add(nb.mul(nb.math('COSINE', nb.mul(am, 4.0)), 0.3), 0.7), 0.042)
        dam = nb.maxf(nb.mapr(nb.sub(dm, lobe), 0.004, -0.002), band(nb, dm, 0.068, 0.0055))
        dam = nb.mul(dam, nb.mul(nb.mapr(rb, 0.3, 0.34), nb.mapr(r, fan.r0 + 0.08, fan.r0 + 0.12)))
        # ---- oxblood lace border
        inb = nb.mapr(rb, 0.29, 0.275)
        edge = nb.mapr(rho, 0.024, 0.016)
        thr = nb.maxf(band(nb, rho, 0.05, 0.007), band(nb, rb, 0.245, 0.006))
        thr = nb.maxf(thr, band(nb, rb, 0.275, 0.005))
        oy = nb.sub(0.2, rho)                                                   # sunburst centre per pleat
        do = nb.math('SQRT', nb.add(nb.mul(s, s), nb.mul(oy, oy)))
        ray = nb.mul(nb.math('ARCTAN2', s, oy), 9.0 / math.pi)
        rays = nb.mul(band(nb, nb.sub(nb.frac(nb.add(ray, 0.5)), 0.5), 0.0, 0.14),
                      nb.mul(nb.mapr(do, 0.04, 0.05), nb.mapr(rho, 0.058, 0.066)))
        eye = nb.maxf(band(nb, do, 0.03, 0.0065), nb.mapr(do, 0.012, 0.008))
        dots = nb.mapr(nb.math('SQRT', nb.add(nb.mul(nb.sub(nb.absf(s), nb.mul(pw, 0.25)), nb.sub(nb.absf(s), nb.mul(pw, 0.25))),
                                               nb.mul(nb.sub(rb, 0.215), nb.sub(rb, 0.215)))), 0.016, 0.011)
        lace = nb.maxf(nb.maxf(edge, thr), nb.maxf(nb.maxf(rays, eye), dots))
        lace = nb.mul(lace, inb)
        net = nb.mul(nb.mul(nb.maxf(band(nb, nb.frac(nb.mul(nb.add(s, rho), 30.0)), 0.5, 0.16),
                                    band(nb, nb.frac(nb.mul(nb.sub(s, rho), 30.0)), 0.5, 0.16)), inb), 0.5)
        pipe = nb.mapr(nb.sub(r, fan.r0), 0.042, 0.034)
        # ---- colour
        c = nb.mixc(nb.mapr(nb.add(mott, nb.mul(fine, 0.3)), 0.3, 0.9), (0.0055, 0.005, 0.0058), (0.012, 0.0105, 0.0115))
        c = nb.mixc(nb.mul(dam, 0.75), c, (0.03, 0.024, 0.026))
        c = nb.mixc(net, c, OXB_D)
        laced = nb.mixc(nb.mapr(nb.add(mott, nb.mul(fine, 0.6)), 0.3, 0.9), OXB_D, OXB)
        c = nb.mixc(nb.maxf(lace, pipe), c, laced)
        wear = nb.mul(nb.math('POWER', fold, 9.0), nb.mapr(nb.add(nb.noise(co, 12.0, 3, 0.6), nb.mul(fine, 0.4)), 0.4, 0.75))
        c = nb.mixc(nb.mul(wear, 0.55), c, (0.05, 0.045, 0.043))
        open_ = nb.mapr(nb.ao(0.05), 0.35, 0.95)
        c = nb.mixc(nb.mul(nb.inv(open_), 0.7), c, (0.002, 0.0018, 0.0018))
        rough = nb.add(nb.mapr(mott, 0.3, 0.7, 0.42, 0.55), nb.mul(lace, 0.25))
        rough = nb.add(nb.sub(rough, nb.mul(dam, 0.2)), nb.mul(wear, 0.25))
        hgt = nb.add(nb.mul(fine, 0.25), nb.add(nb.mul(lace, 0.6), nb.mul(dam, 0.15)))
        return nb.finish(c, rough, 0.0, nb.bump(hgt, 0.3, 0.002))
    return build


def _lacquer(nb):
    """Black lacquer over oxblood undercoat, worn through on the edges, finely crazed."""
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    n1 = nb.noise(co, 18.0, 4, 0.55)
    craze = nb.mul(nb.mapr(nb.voronoi(co, 60.0, 'DISTANCE_TO_EDGE', 'Distance'), 0.02, 0.0),
                   nb.mapr(nb.noise(co, 6.0, 2, 0.5), 0.45, 0.65))
    open_ = nb.mapr(nb.ao(0.03), 0.35, 0.95)
    edge = nb.mul(K.edge_mask(nb, 0.004, 0.03, 0.25), open_)
    wear = nb.mapr(nb.add(edge, nb.mul(nb.noise(co, 40.0, 3, 0.6), 0.45)), 0.6, 1.0)
    c = nb.mixc(nb.mapr(nb.add(n1, nb.mul(rnd, 0.2)), 0.3, 0.8), (0.0035, 0.003, 0.0033), (0.011, 0.0095, 0.01))
    c = nb.mixc(nb.mul(wear, 0.8), c, (0.075, 0.01, 0.011))
    c = nb.mixc(nb.mul(craze, 0.5), c, (0.028, 0.024, 0.022))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.8), c, (0.001, 0.001, 0.001))
    rough = nb.add(nb.mapr(n1, 0.3, 0.7, 0.13, 0.24), nb.add(nb.mul(wear, 0.35), nb.mul(craze, 0.3)))
    return nb.finish(c, rough, 0.0, nb.bump(nb.sub(nb.mul(n1, 0.15), nb.mul(craze, 0.5)), 0.25, 0.002))


def _pearl(nb):
    """Aged pearl: ivory with a warm, uneven nacre and a little pitting, dirty where it sits."""
    co = nb.coord('Object')
    rnd = nb.attr('part_rnd')
    pit = nb.mapr(nb.noise(co, 140.0, 3, 0.5), 0.66, 0.74)
    tint = nb.noise(co, 30.0, 2, 0.5)
    open_ = nb.mapr(nb.ao(0.02), 0.35, 0.95)
    c = nb.mixc(nb.mapr(nb.add(tint, nb.mul(rnd, 0.5)), 0.3, 0.95), (0.6, 0.54, 0.45), (0.36, 0.3, 0.22))
    c = nb.mixc(nb.mul(pit, 0.5), c, (0.16, 0.12, 0.09))
    c = nb.mixc(nb.mul(nb.inv(open_), 0.9), c, (0.03, 0.025, 0.02))
    return nb.finish(c, nb.add(nb.mapr(tint, 0.3, 0.7, 0.18, 0.3), nb.mul(pit, 0.3)), 0.0)


K.RECIPES.update({'lacquer': _lacquer, 'pearl': _pearl})

# ---------------------------------------------------------------------------------------- the pair
FANS = [Fan('L', 1, (0.36, 0.66, 1.62), -14, 136, 16, 24, 4, (1.62, 1.8, 1.4, 0.32), 0.64, 5.8, 0.07,
            tear=(8.35, 10.7, 0.5)),
        Fan('R', -1, (-0.36, 0.66, 1.56), -6, 106, 16, 16, -5, (1.66, 1.74, 1.36, 0.25), 0.6, 5.8, 0.06)]
for f in FANS:
    K.RECIPES['silk_' + f.key] = silk_recipe(f)

VIEWS = [('preview', (0.0, 0.75, 1.75), (0.62, 1.0, 0.3), 8.8, 50),
         ('back', (0.0, 0.7, 1.75), (0.0, 1.0, 0.1), 8.6, 50),
         ('front', (0.0, 0.0, 1.75), (0.0, -1.0, 0.08), 8.6, 50),
         ('closeup', (-0.45, 0.72, 1.42), (-0.5, 1.0, 0.25), 2.3, 50)]

K.ensure_mannequin()
K.new_scene()
A = K.Asm()
bases = {}
for f in FANS:
    f.leaf(A)
    f.sticks(A)
    f.guards(A)
    bases[f.key] = f.pivot(A)

# --- one short chain from the right fan's rivet plate, with a small tarnished puffy heart
fr = FANS[1]
down = K._unit(-Z - fr.N * np.dot(-Z, fr.N))
pb = K._unit(np.cross(Z, fr.N))                 # bail normal: horizontal, across the fan
RB, WB, RJ, WJ = 0.022, 0.0055, 0.019, 0.0048
bc = bases['R'] + fr.N * 0.006 + down * 0.128 - Z * (RB - WB - 0.004)
A.part('chain', K.ring(RB, RB, WB, segs=10, sides=4), 'silver', K.frame(bc, Z, pb))
s0 = bc - Z * RB
off0, off1 = LA - LR - WB, LA - LR - WJ
end = s0 - Z * (off0 + 4 * PITCH + off1)       # five links
for c, t, p in K.chain_frames(K.Path([s0, end]), off0, off1, PITCH, ref=pb, odd=True):
    A.part('chain', K.ring(LA, LB, LR, segs=8, sides=3), 'silver', K.frame(c, t, p))
cj = end - Z * RJ
A.part('chain', K.ring(RJ, RJ, WJ, segs=8, sides=4), 'silver', K.frame(cj, Z, pb))


def heart(P):
    q = P * np.array([1.0, 1 / 0.42, 1.0])
    lobes = np.minimum(S.sphere(q, (-0.031, 0, -0.045), 0.036), S.sphere(q, (0.031, 0, -0.045), 0.036))
    tip = np.minimum(S.capsule(q, (-0.024, 0, -0.058), (0, 0, -0.128), 0.034, 0.005),
                     S.capsule(q, (0.024, 0, -0.058), (0, 0, -0.128), 0.034, 0.005))
    return S.smin(lobes, tip, 0.014)


hv, hf, hu = S.mesh(heart, (-0.075, -0.02, -0.14), (0.075, 0.02, 0.0), 0.0035, 150, sym='X', name='heart')
hm = np.eye(4)
hm[:3, 0], hm[:3, 1], hm[:3, 2], hm[:3, 3] = pb, np.cross(Z, pb), Z, cj - Z * RJ + Z * 0.02
A.part('heart', (hv, hf, hu), 'silver', hm, smooth=60)

body = K.dev_modes(A, NAME, VIEWS)
ob, rep = K.finish_item(A, NAME, 'Back', 'BodyBack', 'Ashen Lace Fan Wings', meta=body)
K.previews(NAME, VIEWS, spp=int(os.environ.get('PREVIEW_SPP', 40)))
print('DONE', NAME)
