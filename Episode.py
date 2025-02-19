import importlib
from multiprocessing import Queue, Process, TimeoutError
from World import World

class Episode(object):
    def __init__(self, args, episode_index, carla_port, traffic_port, selected_gpu, selected_map, selected_weather, recorder_queue):
        self.args = args
        self.carla = None
        self.load_carla()
        self.input_queue = None
        self.output_queue = None
        self.recorder_queue = recorder_queue
        self.episode_index = episode_index
        self.carla_port = carla_port
        self.traffic_port = traffic_port
        self.selected_gpu = selected_gpu
        self.selected_map = selected_map
        self.selected_weather = selected_weather
        self.episode_running = False

    def load_carla(self):
        if self.carla is not None:
            self.carla = importlib.reload(self.carla)
        else:
            self.carla = importlib.import_module("carla")

    @staticmethod
    def _episode_launch_thread(args, episode_index, weather, carla_map, carla_port, selected_gpu, traffic_port, input_queue, output_queue):
        total_ticks = args.switch * args.fps
        world = World(args, episode_index, weather, carla_map, carla_port, traffic_port, selected_gpu, input_queue, output_queue)
        for i in range(total_ticks):
            world.tick()
        world.destroy()
        output_queue.put(None)
        output_queue.close()

    def launch_episode(self):
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.episode_process = Process(target=Episode._episode_launch_thread, args=(self.args, self.episode_index, self.selected_weather, self.selected_map, self.carla_port, self.selected_gpu, self.traffic_port, self.input_queue, self.output_queue))
        self.episode_running = True
        self.episode_process.start()

    def consume_episode_data(self):
        try:
            data = self.output_queue.get(timeout=60)
            if data is None:
                self.episode_running = False
                self.input_queue.close()
                return
            self.recorder_queue.put(data)
        except:
            print("Simulator subprocess appears to have either crashed or otherwise gone silent - Timeout. Will attempt to restart it!")
            self.kill_episode(restart=True)
            self.launch_episode()
    
    def kill_episode(self, restart=False):
        self.episode_process.kill()
        self.episode_running = restart
        self.input_queue = None 
        self.output_queue = None

    def loop(self):
        self.launch_episode()
        while self.episode_running:
            self.consume_episode_data()
        self.kill_episode()
