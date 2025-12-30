import argparse
import csv
import threading
import time
import os
import signal
from math import sqrt, asin, degrees

from krpc.error import StreamError
import krpc
import yaml

from constants import *
from tuple_math import norm, dot, cross
import guidance


# Guidance settings
T_GUIDANCE_MARGIN = 5  # How many seconds past tgo to wait for termination before shutdown

# Settings for complementary filter estimating T, Isp
ENGINE_FILTER_ALPHA = .9
ENGINE_FILTER_MIN_T = 2
ENGINE_FILTER_MIN_N = 10


class LoggingThread(threading.Thread):

    def __init__(self, conn, vessel, tm_dict):
        super().__init__()
        self.log_sample_time = tm_dict['sample_time']
        self.tm_dict = {}
        rf = vessel.orbit.body.reference_frame
        # Function to pack vector TM streams into a single value
        def tm_reader(stream):
            def inner():
                value = stream()
                try:
                    if len(value) > 1:
                        return norm(value)
                except TypeError:
                    pass
                return value
            return inner
        # Gather all TM streams
        for k, v in tm_dict['vessel'].items():
            self.tm_dict[k] = tm_reader(conn.add_stream(getattr, vessel, v))
        for k, v in tm_dict['orbit'].items():
            self.tm_dict[k] = tm_reader(conn.add_stream(getattr, vessel.orbit, v))
        for k, v in tm_dict['flight'].items():
            self.tm_dict[k] = tm_reader(conn.add_stream(getattr, vessel.flight(rf), v))
        self.stopped = False

    def run(self):
        with open('telemetry.csv', 'w', newline='') as f:
            f.write(','.join(self.tm_dict.keys()))
            f.write('\n')
            writer = csv.writer(f)
            while not self.stopped:
                tm_row = [v() for _, v in self.tm_dict.items()]
                writer.writerow(tm_row)
                print(f"|".join(f"{k:>14s}" for k in self.tm_dict.keys()))
                print(f"|".join(f"{v: 14.6f}" for v in tm_row))
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
    
    def __init__(self, conn, vessel, configs):
        super().__init__()
        self.conn = conn
        self.vessel = vessel
        self.fairings_action_group = configs['action_group']
        self.fairings_dynamic_pressure = configs['max_dynamic_pressure']
        self.fairings_minimum_altitude = configs['min_altitude']
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
        self.stopped = True
    
    def stop(self):
        print("Fairing Sequencer Override")
        self.stopped = True


class StagingThread(threading.Thread):
    
    def __init__(self, conn, vessel, ut_stream, config, n_stage):
        super().__init__()
        self.conn = conn
        self.vessel = vessel
        self.ut = ut_stream
        self.config = config
        self.ignited = False
        self.name = config['name']
        if not self.name:
            self.name = f'Stage {n_stage}'
        self.stopped = False
    
    def stage(self, n=1, sleep=1):
        for _ in range(n):
            self.vessel.control.activate_next_stage()
            time.sleep(sleep)

    def run(self):
        thrust = self.conn.get_call(getattr, self.vessel, 'thrust')
        ignition = self.conn.krpc.Expression.greater_than(
            self.conn.krpc.Expression.call(thrust),
            self.conn.krpc.Expression.constant_float(self.config['thrust_threshold']))
        cutoff = self.conn.krpc.Expression.less_than(
            self.conn.krpc.Expression.call(thrust),
            self.conn.krpc.Expression.constant_float(self.config['thrust_threshold']))
        # Pre-coasting
        coast = self.config['pre_coasting']
        if coast:
            print(f"Coasting {coast} s")
            ut0 = self.ut()
            while self.ut() - ut0 < coast:
                time.sleep(1)
        # Stage Start events
        self.stage(n=self.config['start_events'])
        time.sleep(self.config['ullage_time'])
        # Stage Ignition events
        self.vessel.control.throttle = 1.0
        self.stage()  # Engine on
        event = self.conn.krpc.add_event(ignition)
        with event.condition:
            event.wait()
            print(f"{self.name} Ignition")
        self.ignited = True
        self.stage(n=self.config['post_ignition_events'])
        # Stage End events
        event = self.conn.krpc.add_event(cutoff)
        with event.condition:
            event.wait()
            print(f"{self.name} Cutoff")
        self.ignited = False
        time.sleep(1)
        self.stage(n=self.config['cutoff_events'])
        # Post-coasting
        coast = self.config['post_coasting']
        if coast:
            print(f"Coasting {coast} s")
            ut0 = self.ut()
            while self.ut() - ut0 < coast:
                time.sleep(1)
    
    def stop(self):
        print(f"Stopped {self.name} Sequencer")
        self.stopped = True


class Mission():

    def __init__(self, conn, config_file):
        self.conn = conn
        self.vessel = conn.space_center.active_vessel
        self.ut = self.conn.add_stream(getattr, self.conn.space_center, 'ut')
        self.logging_thread = None
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file), 'r') as fp:
            self.configs = yaml.safe_load(fp)
        self.stopped = False
        
    def start_logging(self):
        self.logging_thread = LoggingThread(self.conn, self.vessel, self.configs['log_data'])
        self.logging_thread.start()
        print("Logging started")
    
    def get_isp(self):
        # Needed because it seems that the standard method returns 0
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
        
    def closed_loop_ascent(self, configs, heading):
        # Injection into orbit
        print('Closed loop guidance')
        # Streams
        thrust_stream = self.conn.add_stream(getattr, self.vessel, 'thrust')
        mass_stream = self.conn.add_stream(getattr, self.vessel, 'mass')
        r_vec_stream = self.conn.add_stream(self.vessel.position, self.vessel.orbit.body.non_rotating_reference_frame)
        v_vec_stream = self.conn.add_stream(self.vessel.velocity, self.vessel.orbit.body.non_rotating_reference_frame)
        # Guidance init
        guid: guidance.Guidance = getattr(guidance, configs['type'])(self.vessel, configs, heading, MU_EARTH)
        # Engine parameters filter init
        thrust = thrust_stream()
        Isp = self.get_isp()  # FIXME with streams (why standard stream gets Isp=0?)
        initialized = False
        n_filter = 0
        dt_filter = ENGINE_FILTER_MIN_T / ENGINE_FILTER_MIN_N
        print(f"Average Thrust {thrust/1000:5.1f} kN - Average Isp {Isp:.0f} s")
        # Closed Loop
        while True:
            # Engine parameters filter
            Isp = Isp*ENGINE_FILTER_ALPHA + (1-ENGINE_FILTER_ALPHA)*self.get_isp()
            thrust = thrust*ENGINE_FILTER_ALPHA + (1-ENGINE_FILTER_ALPHA)*thrust_stream()
            if n_filter < ENGINE_FILTER_MIN_N:
                n_filter += 1
                time.sleep(dt_filter)
                continue
            ve = Isp * G0
            tau = ve * mass_stream() / thrust
            # State vector
            t0 = self.ut()
            r0_vec = r_vec_stream()
            v0_vec = v_vec_stream()
            # Guidance call
            res: guidance.Guidance.GuidanceResult = guid(r0_vec, v0_vec, ve, tau)
            if guid.converged:
                try:
                    print(f"  CONVERGED | T: {res.tgo:5.1f} s | P0: {degrees(asin(res.dir[0])):+05.1f} deg")
                except ValueError:
                    continue
                if not initialized:
                    self.vessel.auto_pilot.engage()
                    self.vessel.control.sas = False
                    self.steering_thread = SteeringThread(self.vessel, self.ut, guid.output_reference_frame)
                self.steering_thread.update(t0, res.dir, res.ddir, res.tgo)
                if not initialized:
                    self.steering_thread.start()
                    initialized = True
            else:
                try:
                    print(f"UNCONVERGED | T: {res.tgo:5.1f} s | P0: {degrees(asin(res.dir[0])):+05.1f} deg")
                except ValueError:
                    pass
            time.sleep(1)
            if guid.cutoff:
                break
    
    def terminal_guidance(self, configs):
        # TODO: active guidance rather than just cutoff?
        r_vec_stream = self.conn.add_stream(self.vessel.position, self.vessel.orbit.body.non_rotating_reference_frame)
        v_vec_stream = self.conn.add_stream(self.vessel.velocity, self.vessel.orbit.body.non_rotating_reference_frame)
        Pe_stream = self.conn.add_stream(getattr, self.vessel.orbit, 'periapsis')
        Ap_stream = self.conn.add_stream(getattr, self.vessel.orbit, 'apoapsis')
        # Completion
        dt = .15
        pT = configs['sma'] * (1 - configs['e']**2)
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
        print("Orbital insertion complete")

    def execute(self, log=False):
        # TODO: Support arbitrary number of stages (move per-stage guidance in configs?)
        if log:
            self.start_logging()
        self.steering_thread = None
        self.fairings_thread = None
        try:
            self.fairings_thread = FairingThread(self.conn, self.vessel,
                                                self.configs['vessel']['fairings'])
        except KeyError:
            pass
        # First stage
        self.stage_thread = StagingThread(self.conn, self.vessel, self.ut,
                                          self.configs['vessel']['stages'][0], n_stage=1)
        self.stage_thread.start()
        azimuth = self.configs['guidance']['launch_azimuth']
        self.vertical_ascent(self.configs['guidance']['vertical_ascent']['altitude'],
                             azimuth)
        if self.fairings_thread:
            self.fairings_thread.start()
        self.pitch_program(self.configs['guidance']['pitch_program']['altitude'],
                           self.configs['guidance']['pitch_program']['pitch'],
                           azimuth)
        self.stage_thread.join()
        # Second stage
        if len(self.configs['vessel']['stages']) > 1:
            self.stage_thread = StagingThread(self.conn, self.vessel, self.ut,
                                              self.configs['vessel']['stages'][1], n_stage=2)
            self.stage_thread.start()
            while not self.stage_thread.ignited:
                time.sleep(1)
            time.sleep(1)
            self.closed_loop_ascent(self.configs['guidance']['closed_loop'], azimuth)
            self.terminal_guidance(self.configs['guidance']['closed_loop'])
            self.stage_thread.join()
        # Cleanup
        if self.fairings_thread:
            if not self.fairings_thread.stopped:
                self.fairings_thread.stop()
            self.fairings_thread.join()
        if self.logging_thread:
            self.logging_thread.stop()
            self.logging_thread.join()
        print("Ascent program completed")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('config', type=str)
    parser.add_argument('--address', default='127.0.0.1', type=str)
    parser.add_argument('-l', '--log_data', action='store_true')

    args = parser.parse_args()

    conn = krpc.connect(name='Ascent', address=args.address)
    print("Connected")

    mission = Mission(conn, args.config)
    signal.signal(signal.SIGINT, mission.terminate)

    mission.execute(args.log_data)