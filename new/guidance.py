from abc import ABC, abstractmethod
from collections import namedtuple
from math import sqrt, sin, asin, cos, acos, log, exp, radians

import numpy as np  # Evaluate if we really need it

from constants import *


T_PEG_HARD_CUTOFF = 10


def rv2coe(r, v, mu):
    h = np.cross(r, v)
    n = np.cross(np.array([0, 0, 1]), h)
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    h_mag = np.linalg.norm(h)
    n_mag = np.linalg.norm(n)
    # Eccentricity
    evec = ((v_mag**2 - mu/r_mag)*r - np.dot(r, v)*v) / mu
    e = np.linalg.norm(evec)
    # Semi-major axis
    energy = (v_mag**2)/2 - mu/r_mag
    if abs(e - 1) > 0:
        a = -mu / (2*energy)
    else:
        a = np.inf  # TODO: hyperbolic
    # Inclination
    i = acos(h[-1] / h_mag)
    # Right ascension of the ascending node
    Omega = acos(n[0] / n_mag)
    if n[1] < 0:
        Omega = 2*np.pi - Omega
    # Argument of periapsis
    omega = acos(np.dot(evec, n) / (e*n_mag))
    if evec[-1] < 0:
        omega = 2*np.pi - omega
    # True anomaly
    nu = acos(np.dot(evec, r) / (e*r_mag))
    if np.dot(r, v) < 0:
        nu = 2*np.pi - nu
    return a, e, i, Omega, omega, nu


def rv2ae(r, v, mu):
    r_mag = np.linalg.norm(r)
    v_mag = np.linalg.norm(v)
    evec = ((v_mag**2 - mu/r_mag)*r - np.dot(r, v)*v) / mu
    e = np.linalg.norm(evec)
    energy = (v_mag**2)/2 - mu/r_mag
    if abs(e - 1) > 0:
        a = -mu / (2*energy)
    else:
        a = np.inf  # TODO: hyperbolic
    return a, e


class Guidance(ABC):

    GuidanceResult = namedtuple('GuidanceResult', ['tgo', 'dir', 'ddir'])

    def __init__(self, vessel, configs, heading, mu):
        pass

    @property
    @abstractmethod
    def output_reference_frame(self):
        pass

    @property
    @abstractmethod
    def converged(self):
        pass

    @property
    @abstractmethod
    def cutoff(self):
        pass

    @abstractmethod
    def __call__(self, r0_vec, v0_vec, ve, tau) -> GuidanceResult:
        pass


class PEG(Guidance):

    def __init__(self, vessel, configs, mu):
        self.mu = mu
        nT = 0
        eT = configs['e']
        self.rT = configs['a'] - R_EARTH
        pT = self.rT * (1 + eT*cos(nT))
        self.drT = (MU_EARTH / pT)**.5 * eT * sin(nT)
        self.hT = (MU_EARTH * pT)**.5
        self.wT = self.hT / self.rT**2
        self.hdg = vessel.flight.heading
        self.output_reference_frame = vessel.surface_reference_frame
        self.T = +np.inf
        self.T_cutoff = max(T_PEG_HARD_CUTOFF, configs['tgo_cutoff'])
        self.converged = False
    
    @property
    def cutoff(self):
        return (self.T < self.T_cutoff)

    def __call__(self, r0_vec, v0_vec, ve, tau):
        r0_vec = np.array(r0_vec)
        v0_vec = np.array(v0_vec)
        r0 = np.linalg.norm(r0_vec)
        v0 = np.linalg.norm(v0_vec)
        dr0 = np.dot(r0_vec, v0_vec) / r0
        w0 = (v0**2 - dr0**2)**.5 / r0
        h0 = w0 * r0**2
        for _ in range(1000):
            T = max(0, min(0.995 * tau, self.T))
            veT = ve * T
            veTT = ve * T**2 / 2
            a0 = ve / tau
            aT = a0 / (1 - T/tau)
            b0 = -ve * log(1 - T/tau)
            b1 = b0 * tau - veT
            c0 = b0 * T - b1
            c1 = c0 * tau - veTT
            #
            Ax = self.drT - dr0
            Bx = self.rT - r0 - (dr0 * T)
            detX = b0 * c1 - b1 * c0
            detA = Ax * c1 - b1 * Bx
            detB = b0 * Bx - Ax * c0
            try:
                A = detA / detX
                B = detB / detX
            except ZeroDivisionError:
                break
            C0 = (self.mu / r0**2 - w0**2 * r0) / a0
            CT = (self.mu / self.rT**2 - self.wT**2 * self.rT) / aT
            #
            fr = A + C0
            dfr = B + (CT - C0) / T
            # fh = ??? TODO
            ft = 1 - fr**2 / 2
            dft = -fr * dfr
            ddft = -dfr**2 / 2
            r_avg = (r0 + self.rT) / 2
            Dh = self.hT - h0
            Dv = (((Dh / r_avg) + (veT * (dft + ddft*tau)) + (veTT * ddft)) /
                (ft + dft*tau + ddft*tau**2))
            dT = tau * (1 - exp(-Dv / ve)) - T
            T = T + dT
            if abs(dT / T) < 1e-2:
                self.converged = True
                break
            T = .25*(T) + .75*(T - dT)
        self.T = T
        pitch_0 = asin(A+C0)
        pitch_T = asin(A+CT+B*T)
        dir_0 = (sin(pitch_0), cos(pitch_0)*cos(self.hdg), cos(pitch_0)*sin(self.hdg))
        dir_T = (sin(pitch_T), cos(pitch_T)*cos(self.hdg), cos(pitch_T)*sin(self.hdg))
        ddir = tuple((xt - x0) / T for x0, xt in zip(dir_0, dir_T))
        return self.GuidanceResult(T, dir_0, ddir)


class UPFG(Guidance):

    def __init__(self, vessel, configs, mu):
        self.mu = mu
        self.constraints = self.parse_constraints(configs)
        self.output_reference_frame = vessel.orbit.body.non_rotating_reference_frame
        self.tgo_cutoff = max(T_PEG_HARD_CUTOFF, configs['tgo_cutoff'])
        self.converged = False
    
    @property
    def cutoff(self):
        return (self.tgo < self.tgo_cutoff)
    
    def parse_constraints(self, configs):
        # Convert
        a = configs['a']
        e = configs['e']
        i = configs['i']
        if i is not None:
            i = radians(i)
        O = configs['O']
        if O is not None:
            O = radians(O*15)
        w = configs['w']
        if w is not None:
            w = radians(w)
        tgt_vec = configs['target']
        if all(x is not None for x in tgt_vec):
            tgt_vec = np.array(tgt_vec)
        else:
            tgt_vec = None
        # Select
        bits = tuple(int(x is not None) for x in (a, e, i, O, w, tgt_vec))
        if   bits == (1, 1, 0, 0, 0, 0):
            return self._constraints_1(a, e)
        elif bits == (1, 1, 1, 0, 0, 0):
            return self._constraints_2(a, e, i)
        elif bits == (1, 1, 1, 1, 0, 0):
            return self._constraints_3(a, e, i, O)
        elif bits == (1, 1, 1, 1, 1, 0):
            return self._constraints_4(a, e, i, O, w)
        elif bits == (1, 1, 0, 0, 0, 1):
            return self._constraints_5(a, e, tgt_vec)
        else:
            raise RuntimeError("Invalid combination of constraints")

    def _constraints_1(self, a, e):
        rpe = a * (1-e)
        def inner(rp_vec, vp_vec, iy_vec=None, rd_vec=None):
            if rd_vec is None:
                rd_vec = rp_vec
                rp = np.linalg.norm(rp_vec)
                rd = max(rp, rpe)
                rd_vec = rd/rp * rp_vec
            else:
                rd = np.linalg.norm(rd_vec)
            if iy_vec is None:
                iy_vec = np.cross(vp_vec, rd_vec)
                iy_vec /= np.linalg.norm(iy_vec)
            vd = sqrt(self.mu * (1/rd - 1/a))
            phi = acos(sqrt(self.mu*a*(1-e**2) / (rd*vd)))
            if np.dot(rd_vec, vp_vec) < 0:
                phi = -phi
            vd_vec = vd/rd * (rd_vec*sin(phi) + np.cross(rd_vec, iy_vec)*cos(phi))
            return rd_vec, vd_vec
        return inner
    
    def _constraints_2(self, a, e, i):
        ae_fun = self._constraints_1(a, e)
        def inner(rp_vec, vp_vec):
            iy1_vec = np.cross(vp_vec, rp_vec)
            iy1_vec /= np.linalg.norm(iy1_vec)
            i1 = -acos(iy1_vec[2])
            Di = i - i1
            iy_vec = (iy1_vec*cos(Di) + np.cross(rp_vec, iy1_vec)*sin(Di))
            return ae_fun(rp_vec, vp_vec, iy_vec)
        return inner
    
    def _constraints_3(self, a, e, i, O):
        ae_fun = self._constraints_1(a, e)
        iy_vec = np.array([-sin(i)*sin(O), sin(i)*cos(O), -cos(i)])
        def inner(rp_vec, vp_vec):
            rp = np.linalg.norm(rp_vec)
            rd_vec -= iy_vec * np.dot(iy_vec, rp_vec)
            rd_vec *= rp / np.linalg.norm(rd_vec)
            return ae_fun(rp_vec, vp_vec, iy_vec, rd_vec)
        return inner

    def _constraints_4(self, a, e, i, O, w):
        p = a * (1-e**2)
        ae_fun = self._constraints_1(a, e)
        n_vec = np.array([cos(O), sin(O), 0])
        iy_vec = np.array([-sin(i)*sin(O), sin(i)*cos(O), -cos(i)])
        def inner(rp_vec, vp_vec):
            rp = np.linalg.norm(rp_vec)
            rd_vec -= iy_vec * np.dot(iy_vec, rp_vec)
            rd_vec /= np.linalg.norm(rd_vec)
            l = acos(np.dot(rd_vec, n_vec))
            if np.dot(np.cross(n_vec, rd_vec), iy_vec) > 0:
                l = -l
            rd = p / (1 + e*cos(l-w))
            rd_vec *= rd
            return ae_fun(rp_vec, vp_vec, iy_vec, rd_vec)
        return inner
    
    def _constraints_5(self, a, e, tgt_vec):
        ae_fun = self._constraints_1(a, e)
        def inner(rp_vec, vp_vec):
            iy_vec = np.cross(rp_vec, tgt_vec)
            iy_vec /= np.linalg.norm(iy_vec)
            if np.dot(iy_vec, vp_vec) < 0:
                iy_vec = -iy_vec
            return ae_fun(rp_vec, vp_vec, iy_vec)
        return inner


    def __call__(self, r0_vec, v0_vec, ve, tau, mu, T):
        r0_vec = np.array(r0_vec)
        v0_vec = np.array(v0_vec)
        r0 = np.linalg.norm(r0_vec)
        v0 = np.linalg.norm(v0_vec)
        rG = -(mu*r0_vec/r0**3) / 2  # Assume tgo = 1 s
        # TODO
