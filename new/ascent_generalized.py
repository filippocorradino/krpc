import argparse
import csv
import threading
import time
import signal
from math import sin, asin, tan, atan2, cos, degrees, radians

from krpc.error import StreamError
import krpc

from guidance import peg


G0 = 9.80665
R_EARTH = 6371000
MU_EARTH = 398600000000000
T_PEG_HARD_CUTOFF = 10


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
    
    def __init__(self, conn, vessel, ut_stream, frame):
        super().__init__()
        self.conn = conn
        self.vessel = vessel
        self.ut = ut_stream
        self.vessel.auto_pilot.reference_frame = frame
        self.stopped = False

    def run(self):
        while not self.stopped:
            dt = self.ut() - self.t0
            self.vessel.auto_pilot.target_direction = \
                tuple(d + dt*dd for d, dd in zip(self.dir, self.ddir))
            time.sleep(0.1)
    
    def update(self, t0, dir, ddir, T):
        self.t0 = t0
        self.dir = dir
        self.ddir = ddir
        self.T = T

    def stop(self):
        print("Stopped Steering")
        self.stopped = True


class FairingThread(threading.Thread):
    
    def __init__(self, conn, vessel, args):
        super().__init__()
        self.conn = conn
        self.vessel = vessel
        self.fairings_action_group = args.fairings_action_group
        self.fairings_dynamic_pressure = args.fairings_dynamic_pressure
        self.fairings_minimum_altitude = args.fairings_minimum_altitude
        self.h = self.conn.add_stream(getattr, self.vessel.flight(), 'mean_altitude')
        self.q = self.conn.add_stream(getattr, self.vessel.flight(), 'dynamic_pressure')
        self.stopped = False

    def run(self):
        while not self.stopped:
            h = self.h()
            q = self.q()
            if (h > self.fairings_minimum_altitude and
                q < self.fairings_dynamic_pressure):
                break
            time.sleep(1)
        print(f"Fairing Jett")
        self.vessel.control.set_action_group(self.fairings_action_group, True)
    
    def stop(self):
        print("Stopped Fairing")
        self.stopped = True


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
        self.name = name
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
            print(f"{self.name} Ignition")
        self.ignited = True
        self.stage(n=self.stage_i_events)
        if self.ullage:
            time.sleep(1)
            self.vessel.control.rcs = False
        # Stage End events
        event = self.conn.krpc.add_event(burnout)
        with event.condition:
            event.wait()
            print(f"{self.name} Burnout")
        self.ignited = False
        time.sleep(1)
        self.stage(n=self.stage_e_events)
        if self.endstage:
            print(f"Coasting {self.endstage} s")
            ut = self.conn.add_stream(getattr, self.conn.space_center, 'ut')
            ut0 = ut()
            while ut() - ut0 < self.endstage:
                time.sleep(1)
    
    def stop(self):
        print("Stopped Stage Sequencer")
        self.stopped = True


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
        self.r_vec = self.conn.add_stream(self.vessel.position, self.vessel.orbit.body.non_rotating_reference_frame)
        self.v_vec = self.conn.add_stream(self.vessel.velocity, self.vessel.orbit.body.non_rotating_reference_frame)
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
        self.stopped = False

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
        print('Terminating mission')
        for thread in (self.steering_thread, self.stage_thread,
                       self.fairings_thread, self.logging_thread):
            if thread:
                thread.stop()
                thread.join()
        self.stopped = True
        self.conn.close()  # HACK, cleanly handle waits instead
        exit()

    def vertical_ascent(self, end_altitude, heading):
        while True:
            try:
                start_altitude = self.altitude()
                break
            except StreamError:
                print("Streamerror")
                time.sleep(1)
        vessel_bbox = self.vessel.bounding_box(self.vessel.reference_frame)
        vessel_height = abs(vessel_bbox[0][1] - vessel_bbox[1][1])
        altitude = self.conn.get_call(getattr, self.vessel.flight(), 'mean_altitude')
        liftoff = self.conn.krpc.Expression.greater_than(
            self.conn.krpc.Expression.call(altitude),
            self.conn.krpc.Expression.constant_double(start_altitude+1))
        altitude_target_reached = self.conn.krpc.Expression.greater_than(
            self.conn.krpc.Expression.call(altitude),
            self.conn.krpc.Expression.constant_double(end_altitude))
        tower_clear = self.conn.krpc.Expression.greater_than(
            self.conn.krpc.Expression.call(altitude),
            self.conn.krpc.Expression.constant_double(start_altitude+vessel_height))
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
        event = self.conn.krpc.add_event(tower_clear)
        with event.condition:
            event.wait()
            print("Tower cleared")
        # Linear roll program
        print("Roll program")
        while True:
            altitude = self.altitude()
            k = ((altitude-start_altitude) / (end_altitude-start_altitude))**.5 * 1.1
            roll_cmd =  heading * k + 90 * (1-k)  # Roll program
            if k >= 1:
                break
            self.vessel.auto_pilot.target_pitch_and_heading(90, roll_cmd)
        event = self.conn.krpc.add_event(altitude_target_reached)
        with event.condition:
            event.wait()

    def pitch_program(self, end_altitude, pitch_target, heading):
        # Linear pitch program
        print("Pitch program")
        self.vessel.auto_pilot.engage()
        self.vessel.auto_pilot.target_pitch_and_heading(90, heading)
        start_altitude = self.altitude()
        while True:
            altitude = self.altitude()
            k = ((altitude-start_altitude) / (end_altitude-start_altitude))**.5
            pitch_cmd =  pitch_target * k + 90 * (1-k)  # Pitch program
            if altitude >= end_altitude:
                break
            self.vessel.auto_pilot.target_pitch_and_heading(pitch_cmd, heading)
        # Gravity turn
        print("Gravity turn")
        self.vessel.control.sas = True
        self.vessel.auto_pilot.disengage()
        time.sleep(.1)
        self.vessel.control.sas_mode = self.conn.space_center.SASMode.prograde
        # Wait for staging
        self.stage_thread.join()
        
    def closed_loop_ascent(self, args):
        # Injection into orbit
        print('Closed loop guidance')
        nT = radians(args.tgt_closed_loop_true_anomaly)
        eT = args.tgt_closed_loop_eccentricity
        rT = args.tgt_closed_loop_altitude + R_EARTH
        pT = rT * (1 + eT*cos(nT))
        drT = (MU_EARTH / pT)**.5 * eT * sin(nT)
        hT = (MU_EARTH * pT)**.5
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
            ve = Isp * G0
            tau = ve * self.mass() / thrust
            if not initialized:
                T = 0.995 * tau
            t0 = self.ut()
            r0_vec = self.r_vec()
            v0_vec = self.v_vec()
            A, B, C0, CT, T, converged = peg(r0_vec, v0_vec, hT, rT, drT, ve, tau, MU_EARTH, T)
            if converged:
                try:
                    print(f"  CONVERGED | T: {T:5.1f} s | A: {A:+5.3f} | B: {B:+5.3f} | C: {C0:+5.3f}"
                          f" | P0: {degrees(asin(A + C0)):+05.1f} deg")
                except ValueError:
                    converged = False
                    continue
                if not initialized:
                    self.vessel.auto_pilot.engage()
                    self.vessel.control.sas = False
                    self.steering_thread = SteeringThread(self.conn, self.vessel, self.ut, self.vessel.surface_reference_frame)
                pitch_0 = asin(A+C0)
                pitch_T = asin(A+CT+B*T)
                hdg = radians(args.heading)
                dir_0 = (sin(pitch_0), cos(pitch_0)*cos(hdg), cos(pitch_0)*sin(hdg))
                dir_T = (sin(pitch_T), cos(pitch_T)*cos(hdg), cos(pitch_T)*sin(hdg))
                ddir = tuple((dt - d0) / T for d0, dt in zip(dir_0, dir_T))
                self.steering_thread.update(t0, dir_0, ddir, T)
                if not initialized:
                    self.steering_thread.start()
                    initialized = True
            else:
                try:
                    print(f"UNCONVERGED | T: {T:5.1f} s | A: {A:+5.3f} | B: {B:+5.3f} | C: {C0:+5.3f}"
                          f" | P0: {degrees(asin(A + C0)):+05.1f} deg")
                except ValueError:
                    pass
            time.sleep(1)
            dt = self.ut() - t0
            T = T - dt
            A = A - B * dt
            if (T < args.ref_closed_loop_ttgo and not converged) or T < T_PEG_HARD_CUTOFF:
                break
        # Completion
        dt = .15
        kP = .01  # rad/s
        PeT = pT / (1+eT) - R_EARTH
        ApT = pT / (1-eT) - R_EARTH
        while True:
            t1 = self.ut()
            Pe = self.Pe()
            Ap = self.Ap()
            time.sleep(dt)
            t2 = self.ut()
            if t2-t1 == 0:
                continue
            nPe = self.Pe()
            nAp = self.Ap()
            dr0 = self.vspeed()
            r0 = self.altitude() + R_EARTH
            w0 = (self.ospeed()**2 - dr0**2)**.5 / r0
            h0 = w0 * r0**2
            if h0 >= hT:
                self.steering_thread.update(t0, (0, 0, 0), (0, 0, 0), 0)
                self.vessel.control.throttle = 0.0
                print("Target reached")
                break
            dPe = (nPe-Pe) / (t2-t1)
            dAp = (nAp-Ap) / (t2-t1)
            ttPe = (PeT - nPe) / dPe
            ttAp = (ApT - nAp) / dAp
            P = asin(self.steering_thread.dir[0])
            if (ttPe-ttAp) * dr0 * (rT - nAp) > 0:
                P -= kP * (t2-t1)  # Pitch down if Pe lagging and behind or leading and ahead 
            else:
                P += kP * (t2-t1)  # Pitch up if Pe lagging and ahead or leading and behind
            # self.steering_thread.update(t2, sin(P), 0, 0, 10)
            print(f" FINALIZING "
                  f" | ttPe {max(-99.9,min(99.9,ttPe)):5.1f} s "
                  f" | ttAp {max(-99.9,min(99.9,ttAp)):5.1f} s"
                  f" | Pe {self.Pe()-R_EARTH:8.0f} km | Ap {self.Ap()-R_EARTH:8.0f} km"
                  f" | P0: {degrees(P):+05.1f} deg")
        time.sleep(.1)
        self.steering_thread.stop()
        self.steering_thread.join()
        # self.vessel.control.throttle = 0.0
        self.vessel.auto_pilot.disengage()
        print("Orbital insertion complete!")

    def execute(self, args):
        self.start_logging(args.log_data, args.log_sample_time)
        self.steering_thread = None
        # First stage
        self.stage_thread = StagingThread(self.conn, self.vessel, args, n_stage=1)
        self.stage_thread.start()
        self.vertical_ascent(args.ref_vert_ascent_altitude, args.heading)
        self.pitch_program(args.ref_pitch_progr_altitude, args.ref_pitch_progr_end_pitch, args.heading)
        # Second stage
        self.fairings_thread = FairingThread(self.conn, self.vessel, args)
        self.fairings_thread.start()
        self.stage_thread = StagingThread(self.conn, self.vessel, args, n_stage=2)
        self.stage_thread.start()
        while not self.stage_thread.ignited:
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
    parser.add_argument('-hdg', '--heading', default=90, type=float)
    parser.add_argument('-hpp', '--ref_pitch_progr_altitude', default=7000, type=float)
    parser.add_argument('-ppp', '--ref_pitch_progr_end_pitch', default=70, type=float)
    parser.add_argument('-htg', '--tgt_closed_loop_altitude', default=155000, type=float)
    parser.add_argument('-ntg', '--tgt_closed_loop_true_anomaly', default=0, type=float)
    parser.add_argument('-etg', '--tgt_closed_loop_eccentricity', default=0, type=float)
    parser.add_argument('-tgo', '--ref_closed_loop_ttgo', default=25, type=float)
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