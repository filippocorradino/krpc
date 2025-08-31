from math import sin, cos, acos, degrees

da = 100
de = 0.001
di = 0.001
MU = 3.986E14

def ph(a, e):
    p = a * (1-e*e)
    return p, (p*MU)**.5

def qa(a, aT, e):
    da = a - aT
    sa = (1 + ((a/aT-1) / 3)**4)**.5
    axx = 2 * ((a**3) * (1+e) / (1-e) / MU)**.5
    return sa * (da/axx)**2

def qe(e, eT, a):
    p, h = ph(a, e)
    de = e - eT
    exx = 2*p/h
    return (de/exx)**2

def qi(i, iT, a, e, w):
    p, h = ph(a, e)
    di = acos(cos(i - iT))
    ixx = p/h / ((1-(e*sin(w))**2)**.5 - e*abs(cos(w)))
    return (di/ixx)**2

def qlaw(a, e, i, O, w, nu, aT, eT=None, iT=None):
    q = qa(a, aT, e) + qe(e, eT, a) + qi(i, iT, a, e, w)
    dq_da = (qa(a+da, aT, e) - qa(a, aT, e)) / da
    dq_de = (qe(e+de, eT, a) - qe(e, eT, a)) / de
    dq_di = (qi(i+di, iT, a, e, w) - qi(i, iT, a, e, w)) / di
    p, h = ph(a, e)
    r = p / (1 + e*cos(nu))
    dadot_dfr = 2*a*a/h * e*sin(nu)
    dadot_dft = 2*a*a/h * p/r
    dedot_dfr = 1/h * p*sin(nu)
    dedot_dft = 1/h * ((p+r)*cos(nu) + r*e)
    didot_dfh = r*cos(w+nu) / h
    s_alpha = dq_da*dadot_dfr + dq_de*dedot_dfr
    c_alpha = dq_da*dadot_dft + dq_de*dedot_dft
    alpha = s_alpha / c_alpha
    s_beta = dq_di*didot_dfh
    c_beta = sin(alpha)*s_alpha + cos(alpha)*c_alpha
    beta = s_beta / c_beta
    return q, degrees(alpha), degrees(beta)
