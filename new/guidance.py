import threading
import time
from math import sin, cos, acos, atan, log, exp

import numpy as np


G0 = 9.80665
R_EARTH = 6371000
MU_EARTH = 398600000000000
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
        a = np.inf
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
        a = np.inf
    return a, e


def peg(r0_vec, v0_vec,
        hT, rT, drT,
        ve, tau, mu,
        T):
    r0_vec = np.array(r0_vec)
    v0_vec = np.array(v0_vec)
    r0 = np.linalg.norm(r0_vec)
    v0 = np.linalg.norm(v0_vec)
    dr0 = np.dot(r0_vec, v0_vec) / r0
    w0 = (v0**2 - dr0**2)**.5 / r0
    wT = hT / rT**2
    h0 = w0 * r0**2
    converged = False
    Dtheta = T * (w0 + wT) / 2
    for _ in range(1000):
        T = max(0, min(0.995 * tau, T))
        veT = ve * T
        veTT = ve * T**2 / 2
        a0 = ve / tau
        aT = a0 / (1 - T/tau)
        b0 = -ve * log(1 - T/tau)
        b1 = b0 * tau - veT
        c0 = b0 * T - b1
        c1 = c0 * tau - veTT
        #
        Ax = drT - dr0
        Bx = rT - r0 - (dr0 * T)
        detX = b0 * c1 - b1 * c0
        detA = Ax * c1 - b1 * Bx
        detB = b0 * Bx - Ax * c0
        try:
            A = detA / detX
            B = detB / detX
        except ZeroDivisionError:
            break
        C0 = (mu / r0**2 - w0**2 * r0) / a0
        CT = (mu / rT**2 - wT**2 * rT) / aT
        #
        fr = A + C0
        dfr = B + (CT - C0) / T
        # fh = ??? TODO
        ft = 1 - fr**2 / 2
        dft = -fr * dfr
        ddft = -dfr**2 / 2
        r_avg = (r0 + rT) / 2
        Dh = hT - h0
        Dv = (((Dh / r_avg) + (veT * (dft + ddft*tau)) + (veTT * ddft)) /
            (ft + dft*tau + ddft*tau**2))
        dT = tau * (1 - exp(-Dv / ve)) - T
        T = T + dT
        if abs(dT / T) < 1e-2:
            converged = True
            break
        T = .25*(T) + .75*(T - dT)
    return A, B, C0, CT, T, converged


class UPFG():

    def __init__(self, vessel, ve, tau, mu, rD_vec, vD_vec):
        self.vessel = vessel
        self.ve = ve
        self.tau = tau
        self.mu = mu
        self.rD_vec = rD_vec
        self.vD_vec = vD_vec
        r0_vec = self.r_vec(self.vessel.orbital_reference_frame)
        pass


def upfg(r0_vec, v0_vec, dv_vec,
         rD_vec, vD_vec,
         ve, tau, mu,
         tgo, vgo_vec):
    # TODO: bring vessel sensing in guidance methods
    # TODO: uniform guidance inputs
    r0_vec = np.array(r0_vec)
    v0_vec = np.array(v0_vec)
    rD_vec = np.array(rD_vec)
    vD_vec = np.array(vD_vec)
    r0 = np.linalg.norm(r0_vec)
    rG = -(mu*r0_vec/r0**3) / 2  # Assume tgo = 1 s
    vgo_vec = vD_vec - v0_vec
