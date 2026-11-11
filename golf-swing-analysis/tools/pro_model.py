"""Kinematic 'pro' golfer built on the user's own address skeleton (same body, same setup).

Motion is driven by tour-average style keyframes P1..P10 (pelvis shift/turn, thorax turn
about the spine, spine tilt, lead-arm direction, club direction), interpolated with
smooth (monotone cubic) curves and time-synchronised to the user's own P-positions so
that every frame compares *positions*, not tempo.
"""
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from geom import rot, slerp

D2R = np.pi / 180

# ---------------------------------------------------------------- keyframes
# pelvis: dX toward target, dY up, dZ toward ball (m); close = closed rotation (deg, + = toward trail)
# tilt = lead-hip-up pelvic tilt (deg)
# thorax: turn about the address spine axis (deg, + = closed); lat = extra lean of the spine
# away from target seen face-on (deg); up = forward-bend reduction seen down-the-line (deg)
# head: fraction of thorax turn the head follows
PRO = {
    #        pdx    pdy    pdz   close tilt | turn  lat   up  | head
    "P1":  (0.000, 0.000, 0.000,   0,   0,     0,   0,    0,   0.00),
    "P2":  (-0.010, 0.000, 0.000,  16,   0,    45,   0,    0,   0.25),
    "P3":  (-0.015, -0.005, -0.005, 32,  -2,   72,   1,    1,   0.30),
    "P4":  (-0.020, -0.005, -0.010, 45,  -4,   92,   0,    2,   0.35),
    "P5":  (0.050, -0.012, -0.010,  18,   0,   68,   6,    0,   0.35),
    "P6":  (0.080, -0.010, -0.015, -12,   4,   40,  10,    0,   0.30),
    "P7":  (0.100, 0.000, -0.020, -38,   8,  -22,  15,    3,   0.15),
    "P8":  (0.120, 0.005, -0.020, -58,   8,  -60,  16,    8,  -0.10),
    "P9":  (0.140, 0.010, -0.020, -72,   7,  -85,  13,   14,  -0.35),
    "P10": (0.140, 0.015, -0.030, -92,   5, -112,   7,   26,  -0.60),
}
# lead arm direction (global G frame, unit-ised later) and extension (fraction of arm length)
ARM = {
    "P2": ((-0.55, -0.81, -0.19), 0.99),
    "P3": ((-0.86, -0.02, -0.45), 0.99),
    "P4": ((-0.78, 0.50, -0.38), 0.98),
    "P5": ((-0.92, -0.10, -0.38), 0.99),
    "P6": ((-0.60, -0.78, -0.12), 0.99),
    "P8": ((0.62, -0.78, -0.02), 0.98),
    "P9": ((0.80, 0.20, -0.45), 0.86),
    "P10": ((0.05, 0.75, -0.65), 0.62),
}
# club direction (grip -> clubhead), global
CLUB = {
    "P2": (-0.99, 0.10, 0.06),
    "P3": (-0.25, 0.85, -0.46),
    "P4": (0.90, 0.40, -0.05),
    "P5": (-0.05, 0.85, -0.52),
    "P6": (-0.97, 0.15, -0.10),
    "P8": (0.96, 0.0, -0.28),
    "P9": (0.25, 0.85, -0.45),
    "P10": (-0.55, -0.45, -0.70),
}
# trail heel lift (deg of foot roll about the toe)
HEEL = {"P1": 0, "P5": 0, "P6": 3, "P7": 12, "P8": 40, "P9": 60, "P10": 80}


class ProGolfer:
    def __init__(self, XG, ball, keyframes):
        self.A = {k: np.array(v, float) for k, v in XG.items()}
        A = self.A
        self.kf = keyframes
        self.names = list(keyframes.keys())
        self.tk = np.array([keyframes[k] for k in self.names], float)
        self.ball = np.asarray(ball, float)

        self.P0 = (A[23] + A[24]) / 2
        self.S0 = (A[11] + A[12]) / 2
        self.axis0 = (self.S0 - self.P0) / np.linalg.norm(self.S0 - self.P0)
        self.head0 = (A[7] + A[8]) / 2
        self.nose0 = A[0]
        self.L_up = {s: np.linalg.norm(A[s] - A[s + 2]) for s in (11, 12)}
        self.L_lo = {s: np.linalg.norm(A[s + 2] - A[s + 4]) for s in (11, 12)}
        self.L_th = {h: np.linalg.norm(A[h] - A[h + 2]) for h in (23, 24)}
        self.L_sh = {h: np.linalg.norm(A[h + 2] - A[h + 4]) for h in (23, 24)}
        # grip: lead-hand point and club
        self.grip0 = (A[15] + A[16]) / 2
        self.club_len = np.linalg.norm(self.ball - self.grip0) + 0.02
        self.club0 = (self.ball - self.grip0) / np.linalg.norm(self.ball - self.grip0)
        self.arm0 = (A[15] - A[11]) / np.linalg.norm(A[15] - A[11])
        self.arm_len = self.L_up[11] + self.L_lo[11]
        self.ext0 = np.linalg.norm(A[15] - A[11]) / self.arm_len

        rows = np.array([PRO[k] for k in self.names], float)
        # tour data are real metres; the MediaPipe-scaled body is a bit smaller than life
        leg = np.mean([self.L_th[h] + self.L_sh[h] for h in (23, 24)])
        self.scale = float(np.clip(leg / 0.86, 0.75, 1.15))
        rows[:, :3] *= self.scale
        self.curves = [PchipInterpolator(self.tk, rows[:, c]) for c in range(rows.shape[1])]
        self.heel = PchipInterpolator([keyframes[k] for k in HEEL], list(HEEL.values()))

    # ------------------------------------------------------------- helpers
    def _seg(self, t):
        t = float(np.clip(t, self.tk[0], self.tk[-1]))
        i = int(np.searchsorted(self.tk, t, side="right") - 1)
        i = min(i, len(self.tk) - 2)
        return t, i, (t - self.tk[i]) / (self.tk[i + 1] - self.tk[i])

    def _interp_dir(self, table, t, ctx):
        """Slerp between direction keyframes; P1/P7 directions are computed from context."""
        keys = [k for k in self.names if k in table or k in ("P1", "P7")]
        times = np.array([self.kf[k] for k in keys], float)
        t = float(np.clip(t, times[0], times[-1]))
        j = int(min(np.searchsorted(times, t, side="right") - 1, len(times) - 2))
        u = (t - times[j]) / (times[j + 1] - times[j])
        u = u * u * (3 - 2 * u)  # ease
        a, b = ctx(keys[j]), ctx(keys[j + 1])
        da, db = np.array(a[0], float), np.array(b[0], float)
        return slerp(da, db, u), (1 - u) * a[1] + u * b[1]

    @staticmethod
    def _ik(root, end, l1, l2, pole):
        d = end - root
        dist = np.linalg.norm(d)
        dist_c = min(dist, (l1 + l2) * 0.999)
        dn = d / dist
        end = root + dn * dist_c
        a = (l1 * l1 - l2 * l2 + dist_c * dist_c) / (2 * dist_c)
        h = np.sqrt(max(l1 * l1 - a * a, 0))
        pv = pole - dn * (pole @ dn)
        pv /= np.linalg.norm(pv) + 1e-9
        return root + dn * a + pv * h, end

    # ------------------------------------------------------------- pose
    def pose(self, t):
        A = self.A
        pdx, pdy, pdz, close, tilt, turn, lat, up, headf = [c(np.clip(t, self.tk[0], self.tk[-1])) for c in self.curves]
        Y = np.array([0, 1.0, 0])
        # pelvis
        Pc = self.P0 + np.array([pdx, pdy, pdz])
        Rp = rot(Y, -close * D2R) @ rot([0, 0, 1], tilt * D2R)
        hips = {h: Pc + Rp @ (A[h] - self.P0) for h in (23, 24)}
        # thorax: turn about address spine axis, then global lean (face-on) and bend (DTL)
        Rturn = rot(self.axis0, -turn * D2R)
        # sign check: closed turn should move lead shoulder toward the ball (+Z)
        Rtilt = rot([0, 0, 1], lat * D2R) @ rot([1, 0, 0], -up * D2R)
        Rt = Rtilt @ Rturn
        J = {}
        for s in (11, 12):
            J[s] = Pc + Rt @ (A[s] - self.P0)
        neck = Pc + Rt @ (self.S0 - self.P0)
        # head: follows thorax partially (counter-rotates to keep eyes on ball)
        Rh = Rtilt @ rot(self.axis0, -turn * headf * D2R)
        Rhead_pos = Rtilt @ rot(self.axis0, -turn * 0.15 * D2R)
        head = neck + Rhead_pos @ (self.head0 - self.S0)
        nose = head + Rh @ (self.nose0 - self.head0)
        for j, src in ((7, 7), (8, 8)):
            J[j] = head + Rh @ (A[src] - self.head0)
        J[0] = nose
        J["head"] = head
        J["neck"] = neck
        J.update(hips)
        J["pel"] = Pc

        # lead arm
        def arm_ctx(k):
            if k == "P1":
                return self.arm0, self.ext0
            if k == "P7":
                # impact: lead arm points from lead shoulder to impact grip (hands ahead of ball)
                return self._impact_arm()
            return ARM[k]
        if t <= self.tk[0]:
            adir, ext = self.arm0, self.ext0
        else:
            adir, ext = self._interp_dir(ARM, t, arm_ctx)
        grip = J[11] + adir / np.linalg.norm(adir) * ext * self.arm_len
        if not hasattr(self, "_imp"):
            pass

        # club
        def club_ctx(k):
            if k == "P1":
                return self.club0, 0
            if k == "P7":
                return self._impact_club(), 0
            return CLUB[k], 0
        cdir, _ = (self.club0, 0) if t <= self.tk[0] else self._interp_dir(CLUB, t, club_ctx)
        cdir = cdir / np.linalg.norm(cdir)
        J["grip"] = grip
        J["clubhead"] = grip + cdir * self.club_len
        J["butt"] = grip - cdir * 0.06
        # hands on grip
        lead_w = grip
        trail_w = grip + cdir * 0.075
        J[15], J[16] = lead_w, trail_w
        # elbows
        fwd_chest = Rt @ np.array([0, -0.2, 1.0])
        J[13], J[15] = self._ik(J[11], lead_w, self.L_up[11], self.L_lo[11],
                                 Rt @ np.array([0.3, -1.0, -0.3]))
        J[14], J[16] = self._ik(J[12], trail_w, self.L_up[12], self.L_lo[12],
                                 Rt @ np.array([-0.2, -1.0, 0.1]))

        # feet / legs
        heel = float(self.heel(np.clip(t, self.tk[0], self.tk[-1])))
        for side, hip in ((27, 23), (28, 24)):
            ank, hl, toe = A[side].copy(), A[side + 2].copy(), A[side + 4].copy()
            if side == 28:
                # roll the trail foot up onto its toe: keyframed amount, increased automatically
                # until the leg keeps a natural knee flex (no locked, over-stretched trail leg)
                lmax = self.L_th[hip] + self.L_sh[hip]
                a0, h0 = ank.copy(), hl.copy()
                ax = np.cross(toe - hl, Y)
                ang = heel
                while True:
                    Rf = rot(ax, -ang * D2R) if ang > 0 else np.eye(3)
                    yaw = np.clip(ang / 80, 0, 1) * 50
                    Ry = rot(Y, yaw * D2R)
                    ank = toe + Ry @ (Rf @ (a0 - toe))
                    hl = toe + Ry @ (Rf @ (h0 - toe))
                    if np.linalg.norm(J[hip] - ank) <= 0.975 * lmax or ang >= 85 or t <= self.kf["P5"]:
                        break
                    ang += 1.0
                heel = ang
                # still out of reach (wide stance): let the trail foot slide toward the target
                d = J[hip] - ank
                over = np.linalg.norm(d) - 0.975 * lmax
                if over > 0:
                    dh = np.array([d[0], 0, d[2]])
                    dh /= np.linalg.norm(dh) + 1e-9
                    for _ in range(60):
                        if np.linalg.norm(J[hip] - ank) <= 0.975 * lmax:
                            break
                        ank, hl, toe = ank + dh * 0.005, hl + dh * 0.005, toe + dh * 0.005
            pole = Rp @ np.array([0.15 if side == 27 else -0.1, 0, 1.0])
            if side == 28:
                pole = pole + np.array([0.6, 0, 0]) * np.clip((heel - 3) / 30, 0, 1)
            knee, ank2 = self._ik(J[hip], ank, self.L_th[hip], self.L_sh[hip], pole)
            J[side - 2] = knee
            J[side] = ank2
            J[side + 2] = hl
            J[side + 4] = toe
        return J

    # impact targets (computed from the pro's impact body pose)
    def _impact_grip(self):
        # hands ~6 cm ahead of address grip, 3 cm higher, same distance from body
        return self.grip0 + np.array([0.05, 0.03, 0.0])

    def _impact_arm(self):
        t = self.kf["P7"]
        pdx, pdy, pdz, close, tilt, turn, lat, up, _ = [c(t) for c in self.curves]
        Pc = self.P0 + np.array([pdx, pdy, pdz])
        Rt = rot([0, 0, 1], lat * D2R) @ rot([1, 0, 0], -up * D2R) @ rot(self.axis0, -turn * D2R)
        Lsh = Pc + Rt @ (self.A[11] - self.P0)
        g = self._impact_grip()
        d = g - Lsh
        dn = d / np.linalg.norm(d)
        ext = min(np.linalg.norm(d) / self.arm_len, 0.995)
        self._imp_grip = Lsh + dn * ext * self.arm_len
        return dn, ext

    def _impact_club(self):
        if not hasattr(self, "_imp_grip"):
            self._impact_arm()
        d = self.ball - self._imp_grip
        return d / np.linalg.norm(d)


def load_pro(S, cams_XG, ball):
    kf = json.load(open(f"{S}/keyframes.json"))
    return ProGolfer(cams_XG, ball, kf)
