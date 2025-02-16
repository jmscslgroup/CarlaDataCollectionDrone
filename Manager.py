import importlib
from multiprocessing import Queue, Process
import re
from .World import World
from .Recorder import Recorder

def find_weather_presets(carla):
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

class Manager(object):
    def __init__(self, args):
        self.args = args
        self.carla = None
        self.load_carla()
        self.recorder = Recorder()
        self.input_queue = None
        self.output_queue = None
        self._weather_presets = find_weather_presets(self.carla)
        print("Weather presets: ", self._weather_presets)
        #self._maps = [map for map in self.client.get_available_maps() if "Town" in map]
        self._maps = ["Town01", "Town02", "Town03", "Town04"]#, "Town06", "Town07"]
        self._map_index = 0
        self.episode_runs = 0
        self.episode_max = int(args.total / args.switch)
        self.episode_running = False

    def load_carla(self):
        if self.carla is not None:
            self.carla = importlib.reload(self.carla)
        else:
            self.carla = importlib.import_module("carla")

    @staticmethod
    def _episode_launch_thread(args, weather, carla_map, input_queue, output_queue):
        total_ticks = args.switch * args.fps
        world = World(args, weather, carla_map, input_queue, output_queue)
        for i in range(total_ticks):
            world.tick()
        world.destroy()
        output_queue.put(None)
        output_queue.close()

    def launch_episode(self):
        self.input_queue = Queue()
        self.output_queue = Queue()
        preset = random.choice(self._weather_presets)
        carla_map = self._maps[self._map_index]
        self.episode_process = Process(target=Manager._episode_launch_thread, args=(self.args, preset, carla_map, self.input_queue, self.output_queue))
        self.episode_running = True
        self.episode_process.start()

    def consume_episode_data(self):
        data = self.output_queue.get()
        if data is None:
            self.episode_running = False
            self.input_queue.close()
            return
        array, objs = data
        self.recorder.record_entry(array, objs)
    
    def kill_episode(self):
        self.episode_process.kill()
        self.episode_running = False
        self.input_queue = None 
        self.output_queue = None
        self.episode_runs += 1
        self._map_index += 1
        self._map_index %= len(self._maps)

    def loop(self):
        while self.episode_runs < self.episode_max:
            self.launch_episode()
            while self.episode_running:
                self.consume_episode_data()
            self.kill_episode()
            self.recorder.new_video()
