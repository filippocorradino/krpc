import argparse
import csv
import threading
import time
import signal
from math import sqrt, sin, asin, cos, degrees, radians

from krpc.error import StreamError
import krpc

from guidance import peg


# Physical constants
G0 = 9.80665
R_EARTH = 6371000
MU_EARTH = 398600000000000

# Guidance settings
T_PEG_HARD_CUTOFF = 10  # TODO: bring in guidance
T_GUIDANCE_MARGIN = 5  # How many seconds past tgo to wait for termination before shutdown

# Settings for complementary filter estimating T, Isp
ENGINE_FILTER_ALPHA = .9
ENGINE_FILTER_MIN_T = 2
ENGINE_FILTER_MIN_N = 10


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))

def cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


class LoggingThread(threading.Thread):

    def __init__(self, conn, vessel, log_sample_time, tm_dict):
        super().__init__()
        self.log_sample_time = log_sample_time
        self.tm_dict = {}
        rf = vessel.orbit.body.reference_frame
        for k, v in tm_dict['vessel']:
            self.tm_dict[k] = conn.add_stream(getattr, vessel, v)
        for k, v in tm_dict['orbit']:
            self.tm_dict[k] = conn.add_stream(getattr, vessel.orbit, v)
        for k, v in tm_dict['flight']:
            self.tm_dict[k] = conn.add_stream(getattr, vessel.flight(rf), v)
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
    
    def __init__(self, vessel, ut_stream, frame):
        super().__init__()
        self.vessel = vessel
        self.ut = ut_stream
        self.vessel.auto_pilot.reference_frame = frame
        self.stopped = False

    def run(self):
        while not self.stopped:
            dt = self.ut() - self.t0
            if dt > self.T + T_GUIDANCE_MARGIN:
                break
            self.vessel.auto_pilot.target_direction = \
                tuple(d + dt*dd for d, dd in zip(self.dir, self.ddir))
            time.sleep(0.1)
        self.stop()
    
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
        max_q = False
        q = 0
        while not self.stopped:
            h = self.h()
            dq = self.q() - q
            q += dq
            if (h > self.fairings_minimum_altitude and
                q < self.fairings_dynamic_pressure):
                break
            if not max_q and dq < 0:
                print("Max Q")
                max_q = True
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
        self.logging_thread = None
        self.stopped = False

    def start_logging(self, log_sample_time, tm_dict):
        self.logging_thread = LoggingThread(self.conn, self.vessel, log_sample_time, tm_dict)
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
        altitude_stream = self.conn.add_stream(getattr, self.vessel.flight(), 'mean_altitude')
        while True:
            try:
                start_altitude = altitude_stream()
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
            k = ((altitude_stream()-start_altitude) / (end_altitude-start_altitude))**.5 * 1.1
            roll_cmd =  heading * k + 90 * (1-k)  # Roll program
            if k >= 1:
                break
            self.vessel.auto_pilot.target_pitch_and_heading(90, roll_cmd)
            time.sleep(.02)
        event = self.conn.krpc.add_event(altitude_target_reached)
        with event.condition:
            event.wait()

    def pitch_program(self, end_altitude, pitch_target, heading):
        altitude_stream = self.conn.add_stream(getattr, self.vessel.flight(), 'mean_altitude')
        # Linear pitch program
        print("Pitch program")
        self.vessel.auto_pilot.engage()
        self.vessel.auto_pilot.target_pitch_and_heading(90, heading)
        start_altitude = altitude_stream()
        while True:
            altitude = altitude_stream()
            k = ((altitude-start_altitude) / (end_altitude-start_altitude))**.5
            pitch_cmd =  pitch_target * k + 90 * (1-k)  # Pitch program
            if altitude >= end_altitude:
                break
            self.vessel.auto_pilot.target_pitch_and_heading(pitch_cmd, heading)
            time.sleep(.02)
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
        # Streams
        thrust_stream = self.conn.add_stream(getattr, self.vessel, 'thrust')
        mass_stream = self.conn.add_stream(getattr, self.vessel, 'mass')
        r_vec_stream = self.conn.add_stream(self.vessel.position, self.vessel.orbit.body.non_rotating_reference_frame)
        v_vec_stream = self.conn.add_stream(self.vessel.velocity, self.vessel.orbit.body.non_rotating_reference_frame)
        # TODO: COMPUTE CONSTRAINTS
        nT = radians(args.tgt_closed_loop_true_anomaly)
        eT = args.tgt_closed_loop_eccentricity
        rT = args.tgt_closed_loop_altitude + R_EARTH
        pT = rT * (1 + eT*cos(nT))
        drT = (MU_EARTH / pT)**.5 * eT * sin(nT)
        hT = (MU_EARTH * pT)**.5
        # Engine parameters filter
        thrust = thrust_stream()
        Isp = self.get_isp()  # FIXME with streams (why standard stream gets Isp=0?)
        initialized = False
        n_filter = 0
        dt_filter = ENGINE_FILTER_MIN_T / ENGINE_FILTER_MIN_N
        print(f"Average Thrust {thrust/1000:5.1f} kN - Average Isp {Isp:.0f} s")
        # Closed Loop
        while True:
            Isp = Isp*ENGINE_FILTER_ALPHA + (1-ENGINE_FILTER_ALPHA)*self.get_isp()
            thrust = thrust*ENGINE_FILTER_ALPHA + (1-ENGINE_FILTER_ALPHA)*thrust_stream()
            if n_filter < ENGINE_FILTER_MIN_N:
                n_filter += 1
                time.sleep(dt_filter)
                continue
            ve = Isp * G0
            tau = ve * mass_stream() / thrust
            if not initialized:
                T = 0.995 * tau
            t0 = self.ut()
            r0_vec = r_vec_stream()
            v0_vec = v_vec_stream()
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
                    self.steering_thread = SteeringThread(self.vessel, self.ut, self.vessel.surface_reference_frame)
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
    
    def terminal_guidance(self):
        # TODO: active guidance rather than just cutoff?
        r_vec_stream = self.conn.add_stream(self.vessel.position, self.vessel.orbit.body.non_rotating_reference_frame)
        v_vec_stream = self.conn.add_stream(self.vessel.velocity, self.vessel.orbit.body.non_rotating_reference_frame)
        Pe_stream = self.conn.add_stream(getattr, self.vessel.orbit, 'periapsis')
        Ap_stream = self.conn.add_stream(getattr, self.vessel.orbit, 'apoapsis')
        # Completion
        dt = .15
        # TODO: COMPUTE CONSTRAINTS
        nT = radians(args.tgt_closed_loop_true_anomaly)
        eT = args.tgt_closed_loop_eccentricity
        rT = args.tgt_closed_loop_altitude + R_EARTH
        pT = rT * (1 + eT*cos(nT))
        hT = (MU_EARTH * pT)**.5
        reached = False
        while True:
            time.sleep(dt)
            if self.steering_thread.stopped:
                break
            h0_vec = cross(r_vec_stream(), v_vec_stream())
            h0 = sqrt(dot(h0_vec, h0_vec))
            if h0 >= hT:
                self.steering_thread.update(0, (0, 0, 0), (0, 0, 0), 0)
                self.vessel.control.throttle = 0.0
                reached = True
                break
            print(f" FINALIZING "
                  f" | Pe {Pe_stream()-R_EARTH:8.0f} m | Ap {Ap_stream()-R_EARTH:8.0f} m")
        if reached:
            print("Target reached")
            print(f"Final orbit: {(Pe_stream()-R_EARTH)/1e3:.0f} km x {(Ap_stream()-R_EARTH)/1e3:.0f} km")
        time.sleep(.1)
        self.steering_thread.stop()
        self.steering_thread.join()
        self.vessel.auto_pilot.disengage()
        print("Orbital insertion complete!")

    def execute(self, args):
        if args.log_data:
            self.start_logging(args.log_sample_time, {})
        self.steering_thread = None
        self.fairings_thread = FairingThread(self.conn, self.vessel, args)
        # First stage
        self.stage_thread = StagingThread(self.conn, self.vessel, args, n_stage=1)
        self.stage_thread.start()
        self.vertical_ascent(args.ref_vert_ascent_altitude, args.heading)
        self.fairings_thread.start()
        self.pitch_program(args.ref_pitch_progr_altitude, args.ref_pitch_progr_end_pitch, args.heading)
        # Second stage
        self.stage_thread = StagingThread(self.conn, self.vessel, args, n_stage=2)
        self.stage_thread.start()
        while not self.stage_thread.ignited:
            time.sleep(1)
        time.sleep(1)
        self.closed_loop_ascent(args)
        self.terminal_guidance()


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