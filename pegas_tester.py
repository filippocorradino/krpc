from math import log, exp, sin, cos, radians

G0 = 9.80665
R_EARTH = 6371000
MU_EARTH = 398600000000000

t0 = 0
tau = 167.4
ve = 2669.7
r0 = 6489914
dr0 = 1534.7
w0 = 0.000763
v0 = r0*w0
h0 = r0*r0*w0

nT = radians(148)
eT = 0.089
rT = 237000 + R_EARTH
pT = rT * (1 + eT*cos(nT))
drT = (MU_EARTH / pT)**.5 * eT * sin(nT)
hT = (MU_EARTH * pT)**.5
wT = hT / rT**2

T = 0.99 * tau
A = 0
B = 0

while True:
    for _ in range(1000):
        T = max(0, min(0.99 * tau, T))
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
        C = (MU_EARTH / r0**2 - w0**2 * r0) / a0
        #
        fr = A + C
        dfr = B + ((MU_EARTH / rT**2 - wT**2 * rT) / aT - fr) / T
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
