import argparse
import csv
import threading
import time
import signal
from math import log, exp, sin, asin, cos, degrees, radians

from krpc.error import StreamError
import krpc


G0 = 9.80665
R_EARTH = 6371000
MU_EARTH = 398600000000000
T_PEG_CUTOFF = 25


class LoggingThread(threading.Thread):

    def __init__(self, log_sample_time, tm_dict):
        super().__init__()
        self.log_sample_time = log_sample_time
        self.tm_dict = tm_dict
        self.stopped = False

    def run(self):
        with open('telemetry.csv', 'w', newline='') as f:
            f.write(','.join(self.tm_dict.keys()))
            f.write('\n')
            writer = csv.writer(f)
            while not self.stopped:
                tm_row = [v() for _, v in self.tm_dict.items()]
                writer.writerow(tm_row)
                print(f", ".join(f'{k}: {v:12.6f}'
                                 for k, v in zip(self.tm_dict.keys(), tm_row)))
                time.sleep(self.log_sample_time)

    def stop(self):
        print("Stopped TM logger")
        self.stopped = True


class SteeringThread(threading.Thread):
    
    def __init__(self, conn, vessel, ut_stream, t0, A, B, C, T):
        super().__init__()
        self.conn = conn
        self.vessel = vessel
        self.ut = ut_stream
        self.t0 = t0
        self.A = A
        self.B = B
        self.C = C
        self.T = T

    def run(self):
        while True:
            dt = self.ut() - self.t0
            if dt >= self.T:
                break
            pitch = asin(self.A + self.B*dt + self.C)
            self.vessel.auto_pilot.target_pitch_and_heading(degrees(pitch), 90)
            time.sleep(0.1)
    
    def update(self, t0, A, B, C, T):
        self.t0 = t0
        self.A = A
        self.B = B
        self.C = C
        self.T = T


class FairingThread(threading.Thread):
    
    def __init__(self, conn, vessel, args, h_stream, q_stream):
        super().__init__()
        self.conn = conn
        self.vessel = vessel
        self.fairings_action_group = args.fairings_action_group
        self.fairings_dynamic_pressure = args.fairings_dynamic_pressure
        self.fairings_minimum_altitude = args.fairings_minimum_altitude
        self.h = h_stream
        self.q = q_stream
        self.stopped = False

    def run(self):
        while True:
            h = self.h()
            q = self.q()
            if (h > self.fairings_minimum_altitude and
                q < self.fairings_dynamic_pressure):
                break
            time.sleep(1)
        print(f"Fairing Jett")
        self.vessel.control.set_action_group(self.fairings_action_group, True)


class StagingThread(threading.Thread):
    
    def __init__(self, conn, vessel, args, n_stage, name=''):
        ix_stage = n_stage-1
        super().__init__()
        self.conn = conn
        self.vessel = vessel
        self.thrust_threshold = args.thrust_thresholds[ix_stage]
        self.stage_s_events = args.stage_s_events[ix_stage]
        self.stage_i_events = args.stage_i_events[ix_stage]
        self.stage_e_events = args.stage_e_events[ix_stage]
        self.ullage = args.ullage_times[ix_stage]
        try:
            self.endstage = args.endstage_times[ix_stage]
        except TypeError:
            self.endstage = 0  # endstage wasn't defined
        self.ignited = False
        if not name:
            name = f'Stage {n_stage}'
        self.name = name + ' '
        self.stopped = False
    
    def stage(self, n=1, sleep=1):
        for _ in range(n):
            self.vessel.control.activate_next_stage()
            time.sleep(sleep)

    def run(self):
        thrust = self.conn.get_call(getattr, self.vessel, 'thrust')
        ignition = self.conn.krpc.Expression.greater_than(
            self.conn.krpc.Expression.call(thrust),
            self.conn.krpc.Expression.constant_float(self.thrust_threshold))
        burnout = self.conn.krpc.Expression.less_than(
            self.conn.krpc.Expression.call(thrust),
            self.conn.krpc.Expression.constant_float(self.thrust_threshold))
        # Stage Start events
        self.stage(n=self.stage_s_events)
        if self.ullage:
            # self.vessel.control.rcs = True
            time.sleep(self.ullage)
        # Stage Ignition events
        self.vessel.control.throttle = 1.0
        self.stage()  # Engine on
        event = self.conn.krpc.add_event(ignition)
        with event.condition:
            event.wait()
            print(f"{self.name}Ignition")
        self.ignited = True
        self.stage(n=self.stage_i_events)
        if self.ullage:
            time.sleep(1)
            self.vessel.control.rcs = False
        # Stage End events
        event = self.conn.krpc.add_event(burnout)
        with event.condition:
            event.wait()
            print(f"{self.name}Burnout")
        self.ignited = False
        time.sleep(1)
        self.stage(n=self.stage_e_events)
        if self.endstage:
            print(f"Coasting {self.endstage} s")
            ut = self.conn.add_stream(getattr, self.conn.space_center, 'ut')
            ut0 = ut()
            while ut() - ut0 < self.endstage:
                time.sleep(1)


class Mission():

    def __init__(self, conn):
        self.conn = conn
        self.vessel = conn.space_center.active_vessel
        self.ut = self.conn.add_stream(getattr, self.conn.space_center, 'ut')
        self.altitude = self.conn.add_stream(getattr, self.vessel.flight(), 'mean_altitude')
        self.pressure = self.conn.add_stream(getattr, self.vessel.flight(), 'static_pressure')
        self.q = self.conn.add_stream(getattr, self.vessel.flight(), 'dynamic_pressure')
        self.aspeed = self.conn.add_stream(getattr, self.vessel.flight(self.vessel.orbit.body.reference_frame), 'speed')
        self.vspeed = self.conn.add_stream(getattr, self.vessel.flight(self.vessel.orbit.body.reference_frame), 'vertical_speed')
        self.ospeed = self.conn.add_stream(getattr, self.vessel.orbit, 'speed')
        self.Pe = self.conn.add_stream(getattr, self.vessel.orbit, 'periapsis')
        self.Ap = self.conn.add_stream(getattr, self.vessel.orbit, 'apoapsis')
        self.mach = self.conn.add_stream(getattr, self.vessel.flight(), 'mach')
        self.CD = self.conn.add_stream(getattr, self.vessel.flight(), 'drag_coefficient')
        self.mass = self.conn.add_stream(getattr, self.vessel, 'mass')
        self.thrust = self.conn.add_stream(getattr, self.vessel, 'thrust')
        self.Isp = self.conn.add_stream(getattr, self.vessel, 'specific_impulse')
        # self.apoapsis_altitude = self.conn.add_stream(getattr, self.vessel.orbit, 'apoapsis_altitude')
        # self.periapsis_altitude = self.conn.add_stream(getattr, self.vessel.orbit, 'periapsis_altitude')
        # self.time_to_apoapsis = self.conn.add_stream(getattr, self.vessel.orbit, 'time_to_apoapsis')
        # self.pitch = self.conn.add_stream(getattr, self.vessel.flight(), 'pitch')
        self.logging_thread = None

    def start_logging(self, log_data, log_sample_time):
        if log_data:
            tm_dict = {'UT': self.ut,
                       'h': self.altitude,
                       'p': self.pressure,
                       'va': self.aspeed,
                       'vo': self.ospeed,
                       'M': self.mach,
                       'm': self.mass,
                       'T': self.thrust,
                       'Isp': self.Isp,
                       'CD': self.CD
                       }
            self.logging_thread = LoggingThread(log_sample_time, tm_dict)
            self.logging_thread.start()
            print("Logging started")

    def get_isp(self):
        active_engines = [e for e in self.vessel.parts.engines
                          if e.active and e.has_fuel]
        thrust = sum(engine.thrust for engine in active_engines)
        fuel_consumption = sum(engine.thrust / engine.specific_impulse
                               for engine in active_engines)
        return thrust / fuel_consumption

    def terminate(self, sig, frame):
        # TODO: Improve and gather all threads!
        print('Terminating mission')
        if self.logging_thread:
            self.logging_thread.stop()
            self.logging_thread.join()
        exit()

    def vertical_ascent(self, end_altitude):
        while True:
            try:
                h0 = self.altitude()
                break
            except StreamError:
                print("Streamerror")
                time.sleep(1)
        altitude = self.conn.get_call(getattr, self.vessel.flight(), 'mean_altitude')
        liftoff = self.conn.krpc.Expression.greater_than(
            self.conn.krpc.Expression.call(altitude),
            self.conn.krpc.Expression.constant_double(h0+1))
        altitude_target_reached = self.conn.krpc.Expression.greater_than(
            self.conn.krpc.Expression.call(altitude),
            self.conn.krpc.Expression.constant_double(end_altitude))
        # Liftoff
        self.vessel.control.sas = True
        self.vessel.auto_pilot.disengage()
        self.vessel.control.sas_mode = self.conn.space_center.SASMode.stability_assist
        event = self.conn.krpc.add_event(liftoff)
        with event.condition:
            event.wait()
            print("Liftoff!")
        # Vertical ascent
        self.vessel.auto_pilot.engage()
        self.vessel.auto_pilot.target_pitch_and_heading(90, 90)
        self.vessel.control.sas = False
        event = self.conn.krpc.add_event(altitude_target_reached)
        with event.condition:
            event.wait()

    def pitch_program(self, end_altitude, pitch_target):
        # Linear pitch program
        print("Pitch program")
        self.vessel.auto_pilot.engage()
        self.vessel.auto_pilot.target_pitch_and_heading(90, 90)
        start_altitude = self.altitude()
        while True:
            altitude = self.altitude()
            k = ((altitude-start_altitude) / (end_altitude-start_altitude))**.5
            pitch_cmd =  pitch_target * k + 90 * (1-k)  # Pitch program
            if altitude >= end_altitude:
                break
            self.vessel.auto_pilot.target_pitch_and_heading(pitch_cmd, 90)
        # Gravity turn
        print("Gravity turn")
        self.vessel.control.sas = True
        self.vessel.auto_pilot.disengage()
        time.sleep(1)
        self.vessel.control.sas_mode = self.conn.space_center.SASMode.prograde
        # Wait for staging
        self.stage_1_thread.join()
        
    def closed_loop_ascent(self, args):
        # Injection into orbit
        print('Closed loop guidance')
        nT = radians(args.tgt_closed_loop_true_anomaly)
        eT = args.tgt_closed_loop_eccentricity
        rT = args.tgt_closed_loop_altitude + R_EARTH
        pT = rT * (1 + eT*cos(nT))
        drT = (MU_EARTH / pT)**.5 * eT * sin(nT)
        hT = (MU_EARTH * pT)**.5
        wT = hT / rT**2
        # Average T, Isp
        Isp = 0
        thrust = 0
        avg_n = 10
        avg_t = 2
        for _ in range(avg_n):
            Isp += self.get_isp() / avg_n
            thrust += self.thrust() / avg_n
            time.sleep(avg_t / avg_n)
        initialized = False
        print(f"Average Thrust {thrust/1000:5.1f} kN - Average Isp {Isp:.0f} s")
        # Closed Loop
        while True:
            t0 = self.ut()
            dr0 = self.vspeed()
            r0 = self.altitude() + R_EARTH
            w0 = (self.ospeed()**2 - dr0**2)**.5 / r0
            h0 = w0 * r0**2
            ve = Isp * G0
            tau = ve * self.mass() / thrust
            # print(f"      DEBUG | tau: {tau:5.1f} | ve: {ve:+6.1f} | r0: {r0:+7.0f} | dr0: {dr0:+5.1f} | w0: {w0:+8.6f}")
            converged = False
            if not initialized:
                T = 0.995 * tau
                A = 0
                B = 0
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
            # TODO: better update of C
            if converged:
                try:
                    print(f"  CONVERGED | T: {T:5.1f} s | A: {A:+5.3f} | B: {B:+5.3f} | C: {C:+5.3f}"
                        f" | P0: {degrees(asin(A + C)):+05.1f} deg")
                except ValueError:
                    converged = False
                    continue
                if not initialized:
                    self.vessel.auto_pilot.engage()
                    self.vessel.control.sas = False
                    self.steering_thread = SteeringThread(self.conn, self.vessel, self.ut,
                                                          t0, A, B, C, T)
                    self.steering_thread.start()
                    initialized = True
                self.steering_thread.update(t0, A, B, C, T)
            else:
                try:
                    print(f"UNCONVERGED | T: {T:5.1f} s | A: {A:+5.3f} | B: {B:+5.3f} | C: {C:+5.3f}"
                          f" | P0: {degrees(asin(A + C)):+05.1f} deg")
                except ValueError:
                    pass
            time.sleep(1)
            dt = self.ut() - t0
            T = T - dt
            A = A - B * dt
            if T < T_PEG_CUTOFF and converged:
                break
        # Completion
        dt = .15
        kP = .01  # rad/s
        self.steering_thread.update(t0, A + C, 0, 0, 10)
        PeT = pT / (1+eT) - R_EARTH
        ApT = pT / (1-eT) - R_EARTH
        while True:
            t1 = self.ut()
            Pe = self.Pe()
            Ap = self.Ap()
            time.sleep(dt)
            t2 = self.ut()
            nPe = self.Pe()
            nAp = self.Ap()
            dr0 = self.vspeed()
            r0 = self.altitude() + R_EARTH
            w0 = (self.ospeed()**2 - dr0**2)**.5 / r0
            h0 = w0 * r0**2
            if h0 >= hT:
                self.steering_thread.update(t0, 0, 0, 0, 0)
                self.vessel.control.throttle = 0.0
                print("Target reached")
                break
            dPe = (nPe-Pe) / (t2-t1)
            dAp = (nAp-Ap) / (t2-t1)
            ttPe = (PeT - nPe) / dPe
            ttAp = (ApT - nAp) / dAp
            P = asin(self.steering_thread.A)
            if (ttPe-ttAp) * dr0 * (rT - nAp) > 0:
                P -= kP * (t2-t1)  # Pitch down if Pe lagging and behind or leading and ahead 
            else:
                P += kP * (t2-t1)  # Pitch up if Pe lagging and ahead or leading and behind
            self.steering_thread.update(t2, sin(P), 0, 0, 10)
            print(f" FINALIZING "
                  f" | ttPe {max(-99.9,min(99.9,ttPe)):5.1f} s "
                  f" | ttAp {max(-99.9,min(99.9,ttAp)):5.1f} s"
                  f" | Pe {self.Pe()-R_EARTH:8.0f} km | Ap {self.Ap()-R_EARTH:8.0f} km"
                  f" | P0: {degrees(P):+05.1f} deg")
            # dt = self.ut() - t0
            # T = max(T, dt+1)  # Keep T until h target reached
            # dr0 = self.vspeed()
            # r0 = self.altitude() + R_EARTH
            # w0 = (self.ospeed()**2 - dr0**2)**.5 / r0
            # h0 = w0 * r0**2
            # C = (MU_EARTH / r0**2 - w0**2 * r0) / a0
            # self.steering_thread.update(t0, A, B, C, T)
            # print(f" FINALIZING | htgo: {(hT-h0)/1e9:5.2f} Gm2/s"
            #       f" | Pe {self.Pe()-R_EARTH:8.0f} km | Ap {self.Ap()-R_EARTH:8.0f} km")
            # if h0 >= hT:
            #     self.steering_thread.update(t0, 0, 0, 0, 0)
            #     print("Target reached")
            #     break
            # time.sleep(.1)
        self.steering_thread.join()
        # self.vessel.control.throttle = 0.0
        self.vessel.auto_pilot.disengage()
        print("Orbital insertion complete!")

    def execute(self, args):
        self.start_logging(args.log_data, args.log_sample_time)
        # First stage
        self.stage_1_thread = StagingThread(self.conn, self.vessel, args, n_stage=1)
        self.stage_1_thread.start()
        self.vertical_ascent(args.ref_vert_ascent_altitude)
        self.pitch_program(args.ref_pitch_progr_altitude, args.ref_pitch_progr_end_pitch)
        # Second stage
        self.fairings_thread = FairingThread(self.conn, self.vessel, args, self.altitude, self.q)
        self.fairings_thread.start()
        self.stage_2_thread = StagingThread(self.conn, self.vessel, args, n_stage=2)
        self.stage_2_thread.start()
        while not self.stage_2_thread.ignited:
            time.sleep(1)
        time.sleep(1)
        self.closed_loop_ascent(args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default='127.0.0.1', type=str)
    parser.add_argument('-tth', '--thrust_thresholds', type=float, nargs=2)
    parser.add_argument('-sse', '--stage_s_events', type=int, nargs=2)
    parser.add_argument('-sie', '--stage_i_events', type=int, nargs=2)
    parser.add_argument('-see', '--stage_e_events', type=int, nargs=2)
    parser.add_argument('-udt', '--ullage_times', type=float, nargs=2)
    parser.add_argument('-edt', '--endstage_times', type=float, nargs=2)
    parser.add_argument('-hva', '--ref_vert_ascent_altitude', default=1000, type=float)
    parser.add_argument('-hpp', '--ref_pitch_progr_altitude', default=7000, type=float)
    parser.add_argument('-ppp', '--ref_pitch_progr_end_pitch', default=70, type=float)
    parser.add_argument('-htg', '--tgt_closed_loop_altitude', default=155000, type=float)
    parser.add_argument('-ntg', '--tgt_closed_loop_true_anomaly', default=0, type=float)
    parser.add_argument('-etg', '--tgt_closed_loop_eccentricity', default=0, type=float)
    parser.add_argument('-tgo', '--ref_closed_loop_ttgo', default=30, type=float)
    parser.add_argument('-fag', '--fairings_action_group', default=None, type=int)
    parser.add_argument('-fmh', '--fairings_minimum_altitude', default=50000, type=float)
    parser.add_argument('-fmq', '--fairings_dynamic_pressure', default=100, type=float)
    parser.add_argument('-sff', '--final_staging', action='store_true')
    parser.add_argument('-l', '--log_data', action='store_true')
    parser.add_argument('--log_sample_time', default=2)

    args = parser.parse_args()

    conn = krpc.connect(name='Ascent', address=args.address)
    print("Connected")

    mission = Mission(conn)
    signal.signal(signal.SIGINT, mission.terminate)

    mission.execute(args)