import argparse
import contextlib
import csv
from collections import namedtuple
from copy import copy
from math import atan2, sin, cos, radians, degrees, exp
from os import devnull

import numpy as np
import pandas as pd


G0 = 9.80665
P0 = 101325
T0 = 298.15
R_AIR = 287
GAMMA = 1.4
SCALE_H = 7700
MACH_CUTOFF_P = 10
DRAG_CUTOFF_ALT = 140000
ATMO_LAPSE_RATE_TROPO = -0.0065
ATMO_LAPSE_RATE_STRAT = +0.0023
ATMO_TROPOPAUSE = 16000
T_TP = T0 + ATMO_LAPSE_RATE_TROPO * ATMO_TROPOPAUSE
R_EARTH = 6371000
MU_EARTH = 398600000000000

MIN_STAGE_ALT = 70000
MAX_PPROG_ALT = 10000
QALPHA_LIM = 5000
TIMESTEP = 1


def norm(x):
    return sum([xk**2 for xk in x])**.5


def adam(function, x0, rate=0.05, tol=1e-3, nsteps=1000,
         dx=None, lbounds=None, ubounds=None,
         b1=0.9, b2=0.999, eps=1e-8):
    if dx is None:
        dx = [abs(.01*xk) for xk in x0]
    dim = len(x0)
    df = [0] * dim
    dg = [0] * dim
    v = [0] * dim
    v_hat = [0] * dim
    s = [0] * dim
    s_hat = [0] * dim

    for n in range(nsteps):
        f = function(x0)
        for k in range(dim):
            with contextlib.redirect_stdout(None):
                x = copy(x0)
                x[k] += dx[k]
                df[k] = (function(x) - f) / dx[k]
            v[k] = b1 * v[k] + (1-b1) * df[k]
            v_hat[k] = (1 / 1-(b1**n)) * v[k]
            s[k] = b2 * s[k] + (1-b2) * df[k]**2
            s_hat[k] = (1 / 1-(b2**n)) * s[k]
            dg[k] = rate * v_hat[k] / (s_hat[k]**.5 + eps)
        
        x0 = [xk - dgk for xk, dgk in zip(x0, dg)]
        if lbounds is not None:
            x0 = [max(lb, xk) for xk, lb in zip(x0, lbounds)]
        if ubounds is not None:
            x0 = [min(ub, xk) for xk, ub in zip(x0, ubounds)]
        if abs(1 - (function(x0) / f)) < (tol * rate) and n > 10:
            break
    if n+1 == nsteps:
        print("Optimization stopped - iteration limit exceeded")
    return x0


def gradient_descent(function, x0, rate=0.05, tol=1e-3, nsteps=1000,
                     dx=None, lbounds=None, ubounds=None):
    if dx is None:
        dx = [abs(.01*xk) for xk in x0]
    dim = len(x0)
    df = [0] * dim
    dg = [0] * dim

    for n in range(nsteps):
        f = function(x0)
        for k in range(dim):
            with contextlib.redirect_stdout(None):
                x = copy(x0)
                x[k] += dx[k]
                df[k] = function(x) - f
                dg[k] = rate * df[k] / dx[k]
        x0 = [xk - dgk for xk, dgk in zip(x0, dg)]
        if lbounds is not None:
            x0 = [max(lb, xk) for xk, lb in zip(x0, lbounds)]
        if ubounds is not None:
            x0 = [min(ub, xk) for xk, ub in zip(x0, ubounds)]
        if abs(1 - (function(x0) / f)) < (tol * rate):
            break
    if n+1 == nsteps:
        print("Optimization stopped - iteration limit exceeded")
    return x0


def eom(t, ts, sv, cv, uv, mv):
    # Atmosphere
    h = sv['vpos']
    p = P0 * exp(-h / SCALE_H)
    if h < DRAG_CUTOFF_ALT:
        if h < ATMO_TROPOPAUSE:
            T = T_TP + ATMO_LAPSE_RATE_TROPO * (h - ATMO_TROPOPAUSE)
        else:
            T = T_TP + ATMO_LAPSE_RATE_STRAT * (h - ATMO_TROPOPAUSE)
        rho = p / R_AIR / T
        v2 = (sv['hspd']**2 + sv['vspd']**2)
        M = (v2 / GAMMA / R_AIR / T)**.5
        q = 0.5 * rho * v2
        fd = q * cv.S * cv.CDM(M) / sv['mass']
        # print(f"{rho} {q} {cv.CDM(M)} {fd}")
    else:
        q = 0
        fd = 0
    # Engine
    if t < ts:
        Isp = cv.Isp_vac - ((p / P0) * (cv.Isp_vac - cv.Isp_atm))
        thrust = cv.mflow * Isp * G0
        ft = thrust / sv['mass']
        sv['mass'] -= cv.mflow * TIMESTEP
    else:
        ft = 0
    # Gravity
    fg = G0 - sv['hspd']**2 / (R_EARTH + sv['hpos'])
    # Propagation
    # TODO: make v, fpa as state variables
    fpa = atan2(sv['vspd'], sv['hspd'])
    pitch = radians(sv['pitch'])
    alpha = pitch - fpa
    sv['hpos'] += sv['hspd'] * TIMESTEP/2
    sv['vpos'] += sv['vspd'] * TIMESTEP/2
    sv['vspd'] += (-fg + ft*sin(pitch) - fd*sin(fpa)) * TIMESTEP
    sv['hspd'] += (ft*cos(pitch) - fd*cos(fpa)) * TIMESTEP
    sv['hpos'] += sv['hspd'] * TIMESTEP/2
    sv['vpos'] += sv['vspd'] * TIMESTEP/2
    # Control
    # TODO: bring out of this function and give control as input
    if sv['vpos'] > uv.h_va:
        if sv['vpos'] < uv.h_pp:
            k = ((sv['vpos']-uv.h_va) / (uv.h_pp-uv.h_va))**.5
            sv['pitch'] =  uv.p_pp * k + 90 * (1-k)  # Pitch program
        else:
            sv['pitch'] = degrees(fpa)  # Gravity turn
    # DV losses
    mv['DV'] += ft * TIMESTEP
    mv['dloss'] += fd * TIMESTEP
    mv['gloss'] += G0 * sin(fpa) * TIMESTEP
    mv['sloss'] += ft * (1-cos(alpha)) * TIMESTEP
    # Errors
    if q * alpha > QALPHA_LIM:
        raise RuntimeError("qalpha limit exceeded")
    return t + TIMESTEP, sv, mv


def simulate_ascent(m0, mflow, ts, Isp_vac, Isp_atm, S, CDM, h0,
                    h_va, h_pp, p_pp,
                    log=False, verb=False):
    ConstVector = namedtuple('ConstVector', 'm0 mflow Isp_vac Isp_atm S CDM')
    ControlVector = namedtuple('ControlVector', 'h_va h_pp p_pp')
    sv = {'hpos': 0, 'vpos': h0, 'hspd': 0, 'vspd': 0.1, 'mass': m0, 'pitch':90}
    mv = {'DV': 0, 'dloss': 0, 'gloss': 0, 'sloss': 0}
    cv = ConstVector(m0, mflow, Isp_vac, Isp_atm, S, CDM)
    uv = ControlVector(h_va, h_pp, p_pp)
    t = 0
    # TODO: clean up logging switch
    logfile = devnull
    if log:
        logfile = f'simulation_{h_va:05.0f}_{h_pp:05.0f}_{p_pp:02.0f}.csv'
    with open(logfile, 'w', newline='') as f:
        f.write(f't,{",".join(sv.keys())},{",".join(mv.keys())}')
        f.write('\n')
        writer = csv.writer(f)
        while (t < ts or sv['vpos'] < MIN_STAGE_ALT) and sv['vspd'] > 0:
            writer.writerow([t] + [v for v in sv.values()] + [v for v in mv.values()])
            t, sv, mv = eom(t, ts, sv, cv, uv, mv)
            if sv['vpos'] < 0:
                break
            if verb:
                print(f"t: {t:5.1f} s - "
                    f"V {sv['vpos']:6.0f} m {sv['vspd']:4.0f} m/s - "
                    f"H {sv['hpos']:6.0f} m {sv['hspd']:4.0f} m/s - "
                    f"P {sv['pitch']:4.1f} deg",
                    end='       ')
                print(f"DV {mv['DV']:4.0f} m/s - Losses:  "
                    f"D {100*mv['dloss']/mv['DV']:4.1f}% / "
                    f"G {100*mv['gloss']/mv['DV']:4.1f}% / "
                    f"S {100*mv['sloss']/mv['DV']:4.1f}% ")
    print(f"\nSimulation ran for: Vertical Ascent until {h_va:5.0f} m + "
            f"Pitch Program until {h_pp:5.0f} m and {p_pp:2.0f} deg")
    losses = mv['gloss'] + mv['dloss'] + mv['sloss']
    tot_dv = mv['DV']
    rf = sv['vpos'] + R_EARTH
    vf = norm([sv['hspd'], sv['vspd']])
    print(f"Solution has {100*losses/tot_dv:.1f}% total losses")
    print(f"Stage burnout at {sv['vpos']:.0f} km altitude and {vf:.0f} m/s")
    # TODO: discern between burnout and sim end if different

    if verb:
        spec_energy = vf**2 / 2 - MU_EARTH / rf
        phi = radians(sv['pitch'])
        a = -MU_EARTH / spec_energy / 2
        h = rf * vf * cos(phi)
        p = h**2 / MU_EARTH
        e = (1 - p/a)**.5
        Ap = p/(1-e) - R_EARTH
        Pe = p/(1+e) - R_EARTH
        print(f"Final orbit: [Ap {Ap/1e3:+6.0f} km] [Pe {Pe/1e3:+6.0f} km]")
        
    return sv, mv


def main(args):
    
    # Input data parsing + precalculations
    df = pd.read_csv(args.ascent_datalogger_file, skipinitialspace=True)
    # Get initial and final masses
    h0 = df['h'].values[0]
    m0 = df['m'].values[0]
    mf = df['m'].values[-1]
    while df['CD'].values[0] == 0:
        df.drop(index=df.index[0], axis=0, inplace=True)
    while df['T'].values[-1] == 0:
        df.drop(index=df.index[-1], axis=0, inplace=True)
    # Performance calculations
    df['mflow'] = df['T'] / df['Isp'] / G0
    if args.verbose:
        print()
        print("Parsed ascent telemetry logfile:")
        print(df)
    mflow = df['mflow'].mean()
    ts = (m0 - mf) / mflow
    Isp_atm = df['Isp'].min()
    Isp_vac = df['Isp'].max()
    print()
    print(f"Wet mass  : {m0/1e3:7.1f} t")
    print(f"Dry mass  : {mf/1e3:7.1f} t")
    print(f"Mass flow : {mflow:7.1f} kg/s")
    print(f"Stage time: {ts:7.1f} s")
    print(f"Isp range : {Isp_atm:3.0f}-{Isp_vac:3.0f} s")
    print()
    _ = input('Proceed? ')
    S = args.reference_area
    print(f"Ref area  : {S:7.3f} m2")

    # while df['CD'].values[-1] == 0:
    #     df.drop(index=df.index[-1], axis=0, inplace=True)
    # while df['Ma'].values[-1] != max(df['Ma'].values):
    #     df.drop(index=df.index[-1], axis=0, inplace=True)
    while df['p'].values[-1] < MACH_CUTOFF_P:
        df.drop(index=df.index[-1], axis=0, inplace=True)
    CDM_x = df['Ma'].values
    CDM_y = df['CD'].values

    def CDM(M):
        return np.interp(M, CDM_x, CDM_y)
    
    h_va = args.ref_vert_ascent_altitude
    h_pp = args.ref_pitch_progr_altitude
    p_pp = args.ref_pitch_progr_end_pitch

    if not args.specific_case:
        vopt = [h_va, h_pp, p_pp]
        adim = [1.e3, 1.e4, 1.e2]
        # llim = [h0, 0, -1]
        # llim = [xk / ad for xk, ad in zip(llim, adim)]
        ulim = [MAX_PPROG_ALT, MAX_PPROG_ALT, 90]
        ulim = [xk / ad for xk, ad in zip(ulim, adim)]

        def optimizable(x):
            x[0] = max(x[0], 0)
            sv, mv = simulate_ascent(m0, mflow, ts, Isp_vac, Isp_atm, S, CDM, h0,
                                     x[0]*adim[0], x[1]*adim[1], x[2]*adim[2],
                                     log=False, verb=False)
            losses = (mv['gloss'] + mv['dloss'] + mv['sloss']) / mv['DV']
            return losses
        
        xopt = adam(optimizable,
                    [v / a for v, a in zip(vopt, adim)],
                    rate=.001, tol=1e-6, nsteps=1000,
                    ubounds=ulim
                    )
        vopt = [x * a for x, a in zip(xopt, adim)]
        h_va, h_pp, p_pp = vopt
    
    _, _ = simulate_ascent(m0, mflow, ts, Isp_vac, Isp_atm, S, CDM, h0,
                           h_va, h_pp, p_pp,
                           log=args.log, verb=args.verbose)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('ascent_datalogger_file', type=argparse.FileType('r'))
    parser.add_argument('reference_area', type=float)
    parser.add_argument('-s', '--specific_case', action='store_true')
    parser.add_argument('-hva', '--ref_vert_ascent_altitude', default=1000, type=float)
    parser.add_argument('-hpp', '--ref_pitch_progr_altitude', default=9000, type=float)
    parser.add_argument('-ppp', '--ref_pitch_progr_end_pitch', default=70, type=float)
    parser.add_argument('-l', '--log', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    # parser.add_argument('--target_orbit_altitude', default=140000, type=int)
    # parser.add_argument('--closed_loop_guidance_p_gain', default=0.0002, type=float)
    # parser.add_argument('--closed_loop_guidance_d_gain', default=0.06, type=float)

    args = parser.parse_args()
    main(args)