import importlib
import os
import time
import socket
import subprocess
import signal
from Traffic import Traffic
from CameraManager import CameraManager

def wait_for_port(host, port, timeout=60, check_interval=1):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError):
            if time.time() - start_time > timeout:
                return False
            time.sleep(check_interval)

def wait_for_port_down(host, port, timeout=60, check_interval=1):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                if time.time() - start_time > timeout:
                    time.sleep(check_interval)
        except (socket.timeout, ConnectionRefusedError):
                return True
    return False

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, args, episode_index, weather, carla_map, carla_port, traffic_port, selected_gpu, input_queue, output_queue):
        self.carla = None
        self.episode_index = episode_index
        self.load_carla()
        self.server_process = None
        self.client = None
        self.world = None
        self.sync = args.sync
        self.args = args
        self.traffic_manager = None
        self.actor_role_name = args.rolename
        self.total_ticks = 0
        self.switch_frequency = args.switch * args.fps
        self.traffic = None
        self.camera_manager = None
        self._gamma = args.gamma
        self.weather = weather
        self.carla_map = carla_map
        self.carla_host = "localhost"
        self.carla_port = carla_port
        self.traffic_port = traffic_port
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.carla_command = "DISPLAY=:1 /home/richarwa/Carla-UE4-Dev-Linux-Shipping/CarlaUE4.sh"
        self.start()

    def load_carla(self):
        if self.carla is not None:
            self.carla = importlib.reload(self.carla)
        else:
            self.carla = importlib.import_module("carla")

    def kill_server(self):
        if self.server_process is None:
            return
        os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
        wait_for_port_down(self.carla_host, self.carla_port)
        print("Server killed!")
        self.server_process = None

    def spawn_server(self):
        self.server_process = subprocess.Popen("{} -fps {} -carla-port={}".format(self.carla_command, self.args.fps, int(self.carla_port)), stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        if wait_for_port(self.carla_host, self.carla_port):
            print("Server available!")
        else:
            print("Server not available!")
            raise Exception("Server not available!")

    def spawn_client(self):
        if self.client is not None:
            del self.client
            self.client = None
        self.client = self.carla.Client(self.carla_host, self.carla_port)
        print("Client connected")
        self.client.set_timeout(30.0)
        print("Timeout set!")

        try:
            self.world = self.client.get_world()
            if self.args.sync:
                original_settings = self.world.get_settings()
                settings = self.world.get_settings()
                if not settings.synchronous_mode:
                    settings.synchronous_mode = True
                settings.fixed_delta_seconds = 1.0/float(self.args.fps)

                self.world.apply_settings(settings)
            else:
                settings = self.world.get_settings()
                settings.fixed_delta_seconds = 1.0/float(self.args.fps)
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
        except:
            print("Restarting client")
            self.spawn_client()
        print("Client setup!")
        
    def spawn_server_and_client(self):
        if self.server_process is not None:
            self.kill_server()
        self.load_carla()
        self.spawn_server()
        self.spawn_client()

    def start(self):
        print("Creating new world!")
        print("Map ", self.carla_map)
        self.spawn_server_and_client()
        self.world = self.client.load_world(self.carla_map, reset_settings=False)
        if self.args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        self.world.set_weather(self.weather[0])
        print("Starting traffic!")
        self.traffic_manager = self.client.get_trafficmanager(self.traffic_port)
        self.traffic_manager.set_synchronous_mode(self.args.sync)
        self.traffic = Traffic(self.carla, self.client, self.traffic_manager, self.args)
        self.traffic.instantiate_traffic()
        print("Traffic spawned!")
        # Set up the sensors.
        print("Recorder on new video!")
        print("Starting camera manager!")
        self.camera_manager = CameraManager(self.carla, self.client, self.episode_index, self.world, self.output_queue, self._gamma, self.args)
        self.camera_manager.create_sensors()
        print("Camera Manager finished!")
 
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        print("New weather: ", self.world.get_weather())

    def tick(self):
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        self.camera_manager.tick()
        self.total_ticks += 1
        if (self.total_ticks % int(self.args.fps) == 0):
            print(self.total_ticks)
        
    def destroy_sensors(self):
        self.camera_manager.destroy()
        self.camera_manager = None

    def destroy(self):
        self.destroy_sensors()
        self.traffic.destroy()
        self.traffic_manager.shut_down()
        self.kill_server()
        print("World has been destroyed!")