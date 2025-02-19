import importlib
from multiprocessing import Queue, Process, TimeoutError
import re
import random
import math
from Recorder import Recorder
from Episode import Episode

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
        self.recorder_queue = Queue()
        self._weather_presets = find_weather_presets(self.carla)
        print("Weather presets: ", self._weather_presets)
        self._maps = ["Town01", "Town02", "Town03", "Town04"]
        self.episode_pool_size = self.args.number_of_simulators
        self.episode_process_pool = []
        self.carla_ports = [i for i in range(2000, 2000 + (self.episode_pool_size * 10), 10)]
        self.traffic_ports = [i for i in range(8000, 8000 + (self.episode_pool_size * 10), 10)]
        self.episode_max = int(args.total / args.switch)
        self.episode_previous_image_index = []

    def load_carla(self):
        if self.carla is not None:
            self.carla = importlib.reload(self.carla)
        else:
            self.carla = importlib.import_module("carla")

    @staticmethod
    def _episode_launch_thread(args, pool_number, episode_indices, weathers, carla_maps, carla_port, traffic_port, recorder_queue):
        for index, weather, carla_map in zip(episode_indices, weathers, carla_maps):
            episode = Episode(args, index, carla_port, traffic_port, 0, carla_map, weather, recorder_queue)
            episode.loop()
            print("Episode ", index, " done!")
        print("Pool ", pool_number, " done!")
        recorder_queue.put((None, None, None, pool_number))

    def _process_data(self):
        try:
            array, objs, video_index, image_or_pool_index = self.recorder_queue.get(timeout=60)
            if array is None:
                self.episode_process_pool[image_or_pool_index][1] = False
                return
            if self.episode_previous_image_index[video_index] > image_or_pool_index:
                self.recorder.reset_video(video_index)
            self.recorder.record_entry(array, objs, video_index, image_or_pool_index)
            self.episode_previous_image_index[video_index] = image_or_pool_index
        except:
            print("Episode subprocesses seem to not be working - and this is probably reflective of an issue with our code. Exiting!")
            self.kill_episodes()
            raise Exception("Episode subprocesses seem to not be working - and this is probably reflective of an issue with our code. Exiting!")

    def get_active_pool_count(self):
        count = 0
        for process, active in self.episode_process_pool:
            if active:
                count += 1
        return count

    def run_episodes(self):
        self.episode_process_pool = []
        self.episode_previous_image_index = [-1 for i in range(self.episode_max)]
        episode_indices = [i for i in range(self.episode_max)]
        weathers = random.choices(self._weather_presets, k=self.episode_max)
        carla_maps = random.choices(self._maps, k=self.episode_max)
        chunk_size = int(math.ceil(self.episode_max / self.episode_pool_size))
        for i in range(self.episode_pool_size):
            chunk_beginning = (i * chunk_size)
            chunk_end = (i * chunk_size) + chunk_size
            selected_indices = episode_indices[chunk_beginning:chunk_end]
            selected_weathers = weathers[chunk_beginning:chunk_end]
            selected_maps = carla_maps[chunk_beginning:chunk_end]
            self.episode_process_pool.append([Process(target=Manager._episode_launch_thread, args=(self.args, i, selected_indices, selected_weathers, selected_maps, self.carla_ports[i], self.traffic_ports[i], self.recorder_queue)), True])
            self.episode_process_pool[i][0].start()
        while self.get_active_pool_count() > 0:
            self._process_data()
        self.kill_episodes()

    def kill_episodes(self):
        for process, active in self.episode_process_pool:
            process.kill()
        self.episode_process_pool = []        