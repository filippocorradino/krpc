import argparse
import csv
import threading
import time
import signal

import krpc


class LoggingThread(threading.Thread):

    def __init__(self, log_sample_time, tm_dict):
        super(LoggingThread, self).__init__()
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
        print('Asking to be stopped')
        self.stopped = True


class Mission():

    def __init__(self, conn):
        self.conn = conn
        self.vessel = conn.space_center.active_vessel
        self.ut = self.conn.add_stream(getattr, self.conn.space_center, 'ut')
        self.altitude = self.conn.add_stream(getattr, self.vessel.flight(), 'mean_altitude')
        self.pressure = self.conn.add_stream(getattr, self.vessel.flight(), 'static_pressure')
        self.speed = self.conn.add_stream(getattr, self.vessel.flight(self.vessel.orbit.body.reference_frame), 'speed')  # ECEF speed
        self.mach = self.conn.add_stream(getattr, self.vessel.flight(), 'mach')
        self.c = self.conn.add_stream(getattr, self.vessel.flight(), 'speed_of_sound')
        self.temperature = self.conn.add_stream(getattr, self.vessel.flight(), 'static_air_temperature')
        self.CD = self.conn.add_stream(getattr, self.vessel.flight(), 'drag_coefficient')
        self.mass = self.conn.add_stream(getattr, self.vessel, 'mass')
        self.thrust = self.conn.add_stream(getattr, self.vessel, 'thrust')
        self.Isp = self.conn.add_stream(getattr, self.vessel, 'specific_impulse')
        self.logging_thread = None

    def start_logging(self, log_data, log_sample_time):
        if log_data:
            tm_dict = {'UT': self.ut,
                       'h': self.altitude,
                       'p': self.pressure,
                       'v': self.speed,
                       'Ma': self.mach,
                       'c': self.c,
                       'T': self.temperature,
                       'm': self.mass,
                       'T': self.thrust,
                       'Isp': self.Isp,
                       'CD': self.CD
                       }
            self.logging_thread = LoggingThread(log_sample_time, tm_dict)
            self.logging_thread.start()
            print("Logging started")

    def terminate(self, sig, frame):
        print('Terminating mission')
        if self.logging_thread:
            self.logging_thread.stop()
            self.logging_thread.join()
        exit()

    def execute(self, args):
        self.start_logging(args.log_data, args.log_sample_time)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default='127.0.0.1', type=str)
    parser.add_argument('-l', '--log_data', action='store_true')
    parser.add_argument('--log_sample_time', default=2)

    args = parser.parse_args()

    conn = krpc.connect(name='Logger', address=args.address)
    print("Connected")

    mission = Mission(conn)
    signal.signal(signal.SIGINT, mission.terminate)

    mission.execute(args)