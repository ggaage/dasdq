"""Procedural geometry for Roblox accessories: closed primitives with per-corner UVs in studs.

Every primitive returns (verts, faces, uvs). UVs are laid out at world scale (1 UV unit = 1 stud) so
packing keeps a uniform texel density; long strips are split into islands every CHUNK studs.
"""
import math

import bpy  # must precede bmesh/mathutils: the bpy module sets up their import paths
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils import geometry as mgeo

CHUNK = 0.6


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


def _rot(v, axis, ang):
    axis = axis / np.linalg.norm(axis)
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def _arc(p):
    p = np.asarray(p, float)
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]


def frames(p, closed=False, up=None):
    """Parallel-transport frames (T, N, B) along a polyline."""
    p = np.asarray(p, float)
    n = len(p)
    if closed:
        t = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    else:
        t = np.empty_like(p)
        t[1:-1] = p[2:] - p[:-2]
        t[0], t[-1] = p[1] - p[0], p[-1] - p[-2]
    t = _unit(t)
    a = np.array(up, float) if up is not None else (np.array([0, 0, 1.0]) if abs(t[0][2]) < 0.9
                                                    else np.array([1.0, 0, 0]))
    if abs(np.dot(_unit(a), t[0])) > 0.999:
        a = np.array([1.0, 0, 0]) if abs(t[0][0]) < 0.9 else np.array([0, 1.0, 0])
    nn = np.zeros_like(p)
    nn[0] = _unit(a - t[0] * np.dot(a, t[0]))
    for i in range(1, n):
        ax = np.cross(t[i - 1], t[i])
        s = np.linalg.norm(ax)
        v = nn[i - 1]
        if s > 1e-9:
            v = _rot(v, ax / s, math.atan2(s, np.dot(t[i - 1], t[i])))
        v = v - t[i] * np.dot(v, t[i])
        nn[i] = v / np.linalg.norm(v)
    if closed:
        ax = np.cross(t[-1], t[0])
        s = np.linalg.norm(ax)
        ve = nn[-1]
        if s > 1e-9:
            ve = _rot(ve, ax / s, math.atan2(s, np.dot(t[-1], t[0])))
        ve = _unit(ve - t[0] * np.dot(ve, t[0]))
        ang = math.atan2(np.dot(np.cross(ve, nn[0]), t[0]), np.dot(ve, nn[0]))
        for i in range(n):
            nn[i] = _rot(nn[i], t[i], ang * i / n)
    return t, nn, np.cross(t, nn)


def _bridge(a, b, sides, ua, ub):
    """Faces + per-corner UVs between two rings. A ring of length 1 is a pole; ua/ub hold sides+1 UVs
    (the last one is the seam duplicate; for a pole, entry k is the apex UV of fan triangle k)."""
    f, u = [], []
    if len(a) == 1 and len(b) == 1:
        return f, u
    for k in range(sides):
        k1 = (k + 1) % sides
        if len(a) == 1:
            f.append([a[0], b[k1], b[k]])
            u.append([ua[k], ub[k + 1], ub[k]])
        elif len(b) == 1:
            f.append([a[k], a[k1], b[0]])
            u.append([ua[k], ua[k + 1], ub[k]])
        else:
            f.append([a[k], a[k1], b[k1], b[k]])
            u.append([ua[k], ua[k + 1], ub[k + 1], ub[k]])
    return f, u


def _ring_uv(u, circ, sides, pole):
    off = 0.5 if pole else 0.0
    return [(u, ((k + off) / sides - 0.5) * circ) for k in range(sides + 1)]


def tube(path, radius, sides=6, closed=False, caps=('flat', 'flat'), phase=0.0, up=None, ry=None, chunk=CHUNK,
         ups=None):
    """Swept tube. `radius` (along N) and `ry` (along B) may be scalars or per-point arrays.
    caps: 'flat' (n-gon) or 'point' (the end ring collapses to the path point).
    ups: optional per-point N directions (e.g. surface normals) instead of parallel transport."""
    p = np.asarray(path, float)
    n = len(p)
    rx = np.full(n, radius, float) if np.isscalar(radius) else np.asarray(radius, float)
    rb = rx if ry is None else (np.full(n, ry, float) if np.isscalar(ry) else np.asarray(ry, float))
    tt, nn, bb = frames(p, closed, up)
    if ups is not None:
        u = np.asarray(ups, float)
        nn = _unit(u - tt * np.sum(u * tt, axis=1, keepdims=True))
        bb = np.cross(tt, nn)
    ang = phase + np.arange(sides) * 2 * np.pi / sides
    ca, sa = np.cos(ang), np.sin(ang)
    verts, rings, pole = [], [], []
    for i in range(n):
        is_pole = not closed and ((i == 0 and caps[0] == 'point') or (i == n - 1 and caps[1] == 'point'))
        if is_pole:
            rings.append([len(verts)])
            verts.append(p[i])
        else:
            base = len(verts)
            verts.extend(p[i] + np.outer(ca * rx[i], nn[i]) + np.outer(sa * rb[i], bb[i]))
            rings.append(list(range(base, base + sides)))
        pole.append(is_pole)
    s = _arc(np.vstack([p, p[:1]]) if closed else p)
    circ = np.pi * (rx + rb)
    faces, uvs = [], []
    for i in range(n if closed else n - 1):
        j = (i + 1) % n
        u0 = math.floor(s[i] / chunk) * chunk
        ci = circ[j] if pole[i] else circ[i]
        cj = circ[i] if pole[j] else circ[j]
        f, u = _bridge(rings[i], rings[j], sides, _ring_uv(s[i] - u0, ci, sides, pole[i]),
                       _ring_uv(s[i + 1] - u0, cj, sides, pole[j]))
        faces += f
        uvs += u
    if not closed:
        for end, kind in ((0, caps[0]), (n - 1, caps[1])):
            if kind == 'flat':
                face = rings[end][::-1] if end == 0 else rings[end]
                faces.append(face)
                uvs.append([(float(np.dot(verts[v] - p[end], nn[end])), float(np.dot(verts[v] - p[end], bb[end])))
                            for v in face])
    return np.array(verts), faces, uvs


def sweep_profile(path, profile, normals, ups, chunk=CHUNK):
    """Closed loop path swept with a closed 2D profile [(u, v)] in the (normal, up) frame."""
    pr = np.asarray(profile, float)
    m = len(pr)
    pv = _arc(np.vstack([pr, pr[:1]]))
    pth = np.asarray(path, float)
    n = len(pth)
    s = _arc(np.vstack([pth, pth[:1]]))
    verts, rings = [], []
    for p, nrm, up in zip(pth, normals, ups):
        base = len(verts)
        verts.extend([p + u * np.asarray(nrm) + v * np.asarray(up) for u, v in pr])
        rings.append(list(range(base, base + m)))
    faces, uvs = [], []
    for i in range(n):
        a, b = rings[i], rings[(i + 1) % n]
        u0 = math.floor(s[i] / chunk) * chunk
        ui, uj = s[i] - u0, s[i + 1] - u0
        for k in range(m):
            k1 = (k + 1) % m
            faces.append([a[k], a[k1], b[k1], b[k]])
            uvs.append([(ui, pv[k]), (ui, pv[k + 1]), (uj, pv[k + 1]), (uj, pv[k])])
    return np.array(verts), faces, uvs


def lathe(profile, segs=12, phase=0.0, twist=0.0):
    """Revolve [(r, z)] (top to bottom) around Z. r == 0 entries become poles; open ends get caps."""
    pr = np.asarray(profile, float)
    t = _arc(pr)
    verts, rings, pole = [], [], []
    for j, (r, z) in enumerate(pr):
        if r <= 1e-9:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
            pole.append(True)
            continue
        base = len(verts)
        a0 = phase + twist * j
        verts.extend([(r * math.cos(a0 + 2 * math.pi * k / segs), r * math.sin(a0 + 2 * math.pi * k / segs), z)
                      for k in range(segs)])
        rings.append(list(range(base, base + segs)))
        pole.append(False)

    def circ(j):
        if not pole[j]:
            return 2 * math.pi * pr[j][0]
        nb = j + 1 if j + 1 < len(pr) else j - 1
        return 2 * math.pi * pr[nb][0]
    faces, uvs = [], []
    for j in range(len(rings) - 1):
        f, u = _bridge(rings[j], rings[j + 1], segs, [(v, uu) for uu, v in _ring_uv(t[j], circ(j), segs, pole[j])],
                       [(v, uu) for uu, v in _ring_uv(t[j + 1], circ(j + 1), segs, pole[j + 1])])
        faces += f
        uvs += u
    vv = np.array(verts, float)
    for end in (0, len(rings) - 1):
        if not pole[end]:
            face = rings[end][::-1] if end == 0 else rings[end]
            faces.append(face)
            uvs.append([(vv[i][0], vv[i][1]) for i in face])
    return vv, faces, uvs


def sphere(r, segs=8, rings=5):
    prof = [(0.0, r)] + [(r * math.sin(math.pi * i / rings), r * math.cos(math.pi * i / rings))
                         for i in range(1, rings)] + [(0.0, -r)]
    return lathe(prof, segs)


def ellipse(a, b, n, z=0.0, phase=0.0):
    return np.array([(a * math.cos(phase + 2 * math.pi * i / n), b * math.sin(phase + 2 * math.pi * i / n), z)
                     for i in range(n)])


def ring(a, b, r, segs=8, sides=4, phase=0.0):
    """Closed torus-like ring (oval when a != b) in the XY plane."""
    return tube(ellipse(a, b, segs), r, sides, closed=True, up=(0, 0, 1), phase=phase)


def box(sx, sy, sz):
    x, y, z = sx / 2, sy / 2, sz / 2
    v = np.array([(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
                  (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)], float)
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    axes = [(0, 1), (0, 1), (0, 2), (1, 2), (0, 2), (1, 2)]
    return v, f, [[(v[i][a], v[i][b]) for i in face] for face, (a, b) in zip(f, axes)]


def point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1:
            inside = not inside
    return inside


def triangulate_region(outline, holes=(), interior=()):
    """Constrained Delaunay triangulation of a 2D region with holes. Returns (verts2d, tris)."""
    pts = [tuple(p) for p in outline]
    loops = [list(range(len(pts)))]
    for h in holes:
        s = len(pts)
        pts += [tuple(p) for p in h]
        loops.append(list(range(s, s + len(h))))
    pts += [tuple(p) for p in interior]
    edges = [(lp[i], lp[(i + 1) % len(lp)]) for lp in loops for i in range(len(lp))]
    vs, _, fs, *_ = mgeo.delaunay_2d_cdt([Vector(p) for p in pts], edges, [loops[0]], 1, 1e-7)
    v2 = np.array([v[:] for v in vs])
    tris = [list(f) for f in fs if not any(point_in_poly(v2[list(f)].mean(0), h) for h in holes)]
    return v2, tris


def solid_from_surface(front, tris, back, uv2d=None, chunk=CHUNK):
    """Closed shell from a triangulated surface (`front`/`back` are matching vertex arrays).
    Front/back get planar UVs from `uv2d`, the rim is unrolled along each boundary loop."""
    front, back = np.asarray(front, float), np.asarray(back, float)
    n = len(front)
    uv2d = np.asarray(front[:, :2] if uv2d is None else uv2d, float)
    faces = [list(t) for t in tris] + [[i + n for i in t[::-1]] for t in tris]
    uvs = [[tuple(uv2d[i]) for i in t] for t in tris] + [[(-uv2d[i][0], uv2d[i][1]) for i in t[::-1]] for t in tris]
    count = {}
    for t in tris:
        for i in range(3):
            key = tuple(sorted((t[i], t[(i + 1) % 3])))
            count[key] = count.get(key, 0) + 1
    nxt = {}
    for t in tris:
        for i in range(3):
            a, b = t[i], t[(i + 1) % 3]
            if count[tuple(sorted((a, b)))] == 1:
                nxt[a] = b
    seen = set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, cur = [], start
        while cur not in seen:
            seen.add(cur)
            loop.append(cur)
            cur = nxt[cur]
        length = 0.0
        for idx, a in enumerate(loop):
            b = nxt[a]
            seg = float(np.linalg.norm(front[b] - front[a]))
            u0 = math.floor(length / chunk) * chunk
            da, db = float(np.linalg.norm(front[a] - back[a])), float(np.linalg.norm(front[b] - back[b]))
            faces.append([b, a, a + n, b + n])
            uvs.append([(length + seg - u0, 0.0), (length - u0, 0.0), (length - u0, da), (length + seg - u0, db)])
            length += seg
    return np.vstack([front, back]), faces, uvs


def prism(outline, depth, holes=(), interior=()):
    """Extrude a 2D polygon (with optional holes) along Z, centred on z=0."""
    v2, tris = triangulate_region(outline, holes, interior)
    top = np.c_[v2, np.full(len(v2), depth / 2)]
    bot = np.c_[v2, np.full(len(v2), -depth / 2)]
    return solid_from_surface(top, tris, bot, v2)


def basis(z_axis, y_hint=(0, 0, 1)):
    """Rotation whose local Z points along z_axis and local Y towards y_hint."""
    z = Vector(z_axis).normalized()
    y = Vector(y_hint)
    if abs(z.dot(y.normalized())) > 0.999:
        y = Vector((0, 1, 0)) if abs(z.y) < 0.9 else Vector((1, 0, 0))
    x = y.cross(z).normalized()
    y = z.cross(x).normalized()
    return Matrix((x, y, z)).transposed()


def mat_trs(loc=(0, 0, 0), rot=None, scale=1.0):
    """4x4 matrix. rot: Matrix/Euler/Quaternion or a (z_axis, y_hint) pair for look-at."""
    if rot is None:
        r = Matrix.Identity(3)
    elif isinstance(rot, tuple) and len(rot) == 2 and hasattr(rot[0], '__len__'):
        r = basis(rot[0], rot[1])
    elif hasattr(rot, 'to_matrix'):
        r = rot.to_matrix()
    else:
        r = Matrix(rot)
    s = Matrix.Diagonal(Vector(scale if hasattr(scale, '__len__') else (scale,) * 3))
    return np.array(Matrix.Translation(Vector(loc)) @ (r @ s).to_4x4())


def axes(o, ex, ey):
    """4x4 with local X along ex, local Y along ey (orthogonalised), Z = X x Y, origin o."""
    ex = _unit(ex)
    ey = _unit(np.asarray(ey, float) - ex * np.dot(ey, ex))
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = ex, ey, np.cross(ex, ey), o
    return m


def frame(c, t, p):
    """4x4 with local X along t, local Z along p (orthogonalised against t), origin c."""
    t = _unit(t)
    p = _unit(np.asarray(p, float) - t * np.dot(p, t))
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = t, np.cross(p, t), p, c
    return m


def xf(m, p):
    return (np.asarray(m) @ np.r_[np.asarray(p, float), 1.0])[:3]


class Path:
    """Polyline parametrised by arc length."""

    def __init__(self, pts):
        self.p = np.asarray(pts, float)
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.p, axis=0), axis=1))]
        self.L = float(self.s[-1])

    def at(self, d):
        d = min(max(d, 0.0), self.L)
        i = min(int(np.searchsorted(self.s, d, side='right')) - 1, len(self.p) - 2)
        f = (d - self.s[i]) / max(self.s[i + 1] - self.s[i], 1e-12)
        return self.p[i] + (self.p[i + 1] - self.p[i]) * f

    def tan(self, d, h=0.012):
        return _unit(self.at(d + h) - self.at(d - h))


def sag(p0, p1, depth, n=48):
    t = np.linspace(0, 1, n + 1)
    pts = np.asarray(p0, float) + np.outer(t, np.asarray(p1, float) - np.asarray(p0, float))
    pts[:, 2] -= depth * 4 * t * (1 - t)   # parabola: close enough to a shallow catenary
    return Path(pts)


def chain_frames(path, off0, off1, pitch, ref=(0.0, 1.0, 0.0), odd=False, even=False):
    """(centre, tangent, plane normal) of interlocked links along `path`, the end links `off0`/`off1` in
    from its ends. Even links lie in the plane of the tangent and `ref`, so they hook a ring whose normal
    is `ref`; odd links are turned 90 degrees. odd=True keeps the link count odd so both end links are
    even and can hook such rings; even=True makes the last link odd (turned) instead."""
    ref = np.asarray(ref, float)
    span = path.L - off0 - off1
    n = max(3, int(round(span / pitch)) + 1)
    if (odd and n % 2 == 0) or (even and n % 2):
        n += 1 if span / (n - 1) > pitch else -1
    links = []
    for k in range(n):
        d = off0 + span * k / (n - 1)
        c, t = path.at(d), path.tan(d)
        p = _unit(np.cross(t, ref))
        if k % 2:
            p = _unit(np.cross(t, p))
        links.append((c, t, p))
    return links


def spline(ctrl, per):
    """Catmull-Rom curve through `ctrl`, `per` samples per span."""
    c = np.asarray(ctrl, float)
    c = np.vstack([2 * c[0] - c[1], c, 2 * c[-1] - c[-2]])
    out = []
    for i in range(1, len(c) - 2):
        p0, p1, p2, p3 = c[i - 1], c[i], c[i + 1], c[i + 2]
        for t in np.linspace(0, 1, per, endpoint=False):
            out.append(0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    out.append(c[-2])
    return np.array(out)


def fillet(poly, r, n=4):
    """Round every corner of a 2D polygon with an n-point arc of radius r."""
    p = np.asarray(poly, float)
    out = []
    for i in range(len(p)):
        a, b, c = p[i - 1], p[i], p[(i + 1) % len(p)]
        u, v = _unit(a - b), _unit(c - b)
        half = math.acos(float(np.clip(np.dot(u, v), -1.0, 1.0))) / 2
        cen = b + _unit(u + v) * (r / math.sin(half))
        t1, t2 = b + u * (r / math.tan(half)), b + v * (r / math.tan(half))
        a1 = math.atan2(t1[1] - cen[1], t1[0] - cen[0])
        da = (math.atan2(t2[1] - cen[1], t2[0] - cen[0]) - a1 + math.pi) % (2 * math.pi) - math.pi
        out += [(cen[0] + r * math.cos(a1 + da * k / (n - 1)), cen[1] + r * math.sin(a1 + da * k / (n - 1)))
                for k in range(n)]
    return out


def inset(poly, d):
    """Offset a convex 2D polygon inwards by d."""
    p = np.asarray(poly, float)
    area = float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    s = 1.0 if area > 0 else -1.0
    lines = []
    for i in range(len(p)):
        t = _unit(p[(i + 1) % len(p)] - p[i])
        lines.append((p[i] + s * np.array([-t[1], t[0]]) * d, t))
    out = []
    for i in range(len(p)):
        (p1, t1), (p2, t2) = lines[i - 1], lines[i]
        k = np.linalg.solve(np.array([t1, -t2]).T, p2 - p1)
        out.append(tuple(p1 + t1 * k[0]))
    return out


def cross_outline(h, w, t, yc, p):
    """Latin cross with pointed ends: top point at the origin, foot at y = -h, crossbar centred at yc."""
    a = t / 2
    return [(0.0, 0.0), (-a, -p), (-a, yc + a), (-w / 2 + p, yc + a), (-w / 2, yc), (-w / 2 + p, yc - a),
            (-a, yc - a), (-a, -h + p), (0.0, -h), (a, -h + p), (a, yc - a), (w / 2 - p, yc - a), (w / 2, yc),
            (w / 2 - p, yc + a), (a, yc + a), (a, -p)]


def _planar_uv(v, face):
    p = v[face]
    nrm = np.zeros(3)
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        nrm += np.cross(a, b)
    ax = int(np.argmax(np.abs(nrm)))
    i, j = [k for k in range(3) if k != ax]
    return [(q[i], q[j]) for q in p]


class Asm:
    """Accumulates closed parts into one mesh with per-face material, part id, smoothing and UVs."""

    def __init__(self):
        self.v, self.f, self.uv, self.m, self.r, self.s, self.mats, self.vf = [], [], [], [], [], [], [], []
        self.nv = 0
        self.nf = []          # face count per added part, parallel to self.v

    def add(self, prim, mat, m=None, rnd=None, smooth=180.0, uvs=1.0, vfac=None):
        """vfac: optional per-vertex scalar, readable in materials as the 'vfac' attribute."""
        v, f = np.asarray(prim[0], float), prim[1]
        uv = prim[2] if len(prim) > 2 else [_planar_uv(v, face) for face in f]
        if m is not None:
            v = (np.c_[v, np.ones(len(v))] @ np.asarray(m).T)[:, :3]
        if mat not in self.mats:
            self.mats.append(mat)
        r = float(np.random.random()) if rnd is None else rnd
        self.v.append(v)
        self.vf.append(np.zeros(len(v)) if vfac is None else np.asarray(vfac, float))
        self.f += [[i + self.nv for i in face] for face in f]
        self.uv += [[(a * uvs, b * uvs) for a, b in fu] for fu in uv]
        self.m += [self.mats.index(mat)] * len(f)
        self.r += [r] * len(f)
        self.s += [smooth] * len(f)
        self.nf.append(len(f))
        self.nv += len(v)
        return self

    def tris(self):
        return sum(len(f) - 2 for f in self.f)

    def transform(self, m):
        """Apply a 4x4 to everything added so far (UVs keep their relative scale)."""
        m = np.asarray(m)
        self.v = [(np.c_[v, np.ones(len(v))] @ m.T)[:, :3] for v in self.v]
        return self

    def build(self, name, material_fn):
        me = bpy.data.meshes.new(name)
        me.from_pydata(np.concatenate(self.v).tolist(), [], self.f)
        me.polygons.foreach_set('material_index', self.m)
        for key, vals in (('part_rnd', self.r), ('smooth_deg', self.s)):
            me.attributes.new(key, 'FLOAT', 'FACE').data.foreach_set('value', vals)
        vf = np.concatenate(self.vf)
        if vf.any():
            me.attributes.new('vfac', 'FLOAT', 'POINT').data.foreach_set('value', vf.tolist())
        me.uv_layers.new(name='UVMap').data.foreach_set('uv', [c for fu in self.uv for p in fu for c in p])
        nf = len(me.polygons)
        me.validate()
        if len(me.polygons) != nf:
            print(f'WARN mesh.validate removed {nf - len(me.polygons)} faces')
        for mat in self.mats:
            me.materials.append(material_fn(mat))
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        lay = bm.faces.layers.float['smooth_deg']
        for e in bm.edges:
            if len(e.link_faces) == 2:
                thr = math.radians(min(f[lay] for f in e.link_faces))
                e.smooth = e.calc_face_angle(0.0) < thr
        for f in bm.faces:
            f.smooth = True
        bm.to_mesh(me)
        bm.free()
        me.attributes.remove(me.attributes['smooth_deg'])
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        return ob


def uv_coverage(ob):
    me = ob.data
    uv = np.empty(len(me.loops) * 2)
    me.uv_layers.active.data.foreach_get('uv', uv)
    uv = uv.reshape(-1, 2)
    tot = 0.0
    for p in me.polygons:
        q = uv[p.loop_start:p.loop_start + p.loop_total]
        tot += 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
    return tot


def select_only(ob):
    """Make `ob` the only selected, active object. Objects removed from bpy.data linger as None
    entries in view_layer.objects (and new ones are missing) until the view layer is synced."""
    vl = bpy.context.view_layer
    vl.update()
    for o in vl.objects:
        if o is not None and o != ob:
            o.select_set(False)
    vl.objects.active = ob
    ob.select_set(True)


def pack_uv(ob, margin=0.004):
    """Pack the procedural UV islands into 0..1 (uniform scale, any rotation, concave fit)."""
    select_only(ob)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    try:
        bpy.ops.uv.select_all(action='SELECT')
    except RuntimeError as exc:   # no UV editor in background mode; packing then uses the face selection
        print('UV select_all skipped:', str(exc).splitlines()[0])
    bpy.ops.uv.pack_islands(rotate=True, rotate_method='ANY', scale=True, margin=margin, shape_method='CONCAVE')
    bpy.ops.object.mode_set(mode='OBJECT')
    cov = uv_coverage(ob)
    print(f'UV packed coverage={cov:.3f}')
    return cov
