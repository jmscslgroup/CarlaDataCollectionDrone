#!/usr/bin/env python

import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import os
import weakref
import time
import cv2
import copy
import shutil
import json
from multiprocessing import Queue, Process, Value, Lock

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

OBJECT_TO_COLOR = [
    (255, 255, 255),
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142,  35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0,  60, 100),
    (0,  80, 100),
    (0, 0, 230),
    (119, 11, 32),
    (110, 190, 160),
    (170, 120, 50),
    (55, 90, 80),
    (45, 60, 150),
    (157, 234, 50),
    (81, 0, 81),
    (150, 100, 100),
    (230, 150, 140),
    (180, 165, 180),
]

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3, 4]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, client, carla_world, hud, args):
        self.client = client
        self.world = carla_world
        self.sync = args.sync
        self.args = args
        self.traffic_manager = None
        self.actor_role_name = args.rolename
        self.total_ticks = 0
        self.switch_frequency = args.switch * args.fps
        self.tick_limit = args.total * args.fps
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.traffic = None
        self.camera_manager = None
        self.recorder = None
        self._weather_presets = find_weather_presets()
        print("Weather presets: ", self._weather_presets)
        #self._maps = [map for map in self.client.get_available_maps() if "Town" in map]
        self._maps = ["Town01", "Town02", "Town03", "Town04", "Town05"]#, "Town06", "Town07"]
        self._map_index = 0
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def restart(self):
        print("Beginning restart!")
        if self.traffic is not None:
            print("Destroying traffic!")
            self.traffic.destroy()
            self.traffic_manager = None
            self.traffic = None
            print("Traffic destroyed!")
        if self.camera_manager is not None:
            print("Destroying sensors!")
            self.destroy_sensors()
            print("Sensors destroyed!")

        print("Creating new world!")
        print("Map ", self._maps[self._map_index])
        self.world = self.client.load_world(self._maps[self._map_index], reset_settings=False)
        print("World created!")
        if self.args.sync:
            print("Doing synchronous tick!")
            self.world.tick()
        else:
            print("Doing async tick!")
            self.world.wait_for_tick()
        print("Tick done!")
        print("Map ", self._maps[self._map_index])
        self._map_index += 1
        self._map_index %= len(self._maps)
        self.next_weather()
        print("Starting traffic!")
        self.traffic_manager = self.client.get_trafficmanager()
        self.traffic_manager.set_synchronous_mode(self.args.sync)
        print("Spawning traffic!")
        self.traffic = Traffic(self.client, self.traffic_manager, self.args)
        self.traffic.instantiate_traffic()
        print("Traffic spawned!")
        # Set up the sensors.
        if self.recorder is None:
            self.recorder = Recorder()
        else:
            self.recorder.new_video()
        print("Recorder on new video!")
        print("Starting camera manager!")
        self.camera_manager = CameraManager(self.client, self.world, self.recorder, self.hud, self._gamma, self.args)
        self.camera_manager.create_sensors()
        print("Camera Manager finished!")
 
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        print("New weather: ", self.world.get_weather())

    def next_weather(self, reverse=False):
        preset = random.choice(self._weather_presets)
        self.world.set_weather(preset[0])
        print("New weather preset: ", preset, preset[0])

    def tick(self, clock):
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        self.hud.tick(self, clock)
        self.camera_manager.tick()
        self.total_ticks += 1
        if (self.total_ticks % int(self.args.fps) == 0):
            print(self.total_ticks)
        if ((self.total_ticks % self.switch_frequency) == 0):
            print("Restarting at ", self.total_ticks, " with frequnecy of ", self.switch_frequency)
            self.restart()
        if (self.total_ticks >= self.tick_limit):
            self.destroy()
        
    def destroy_sensors(self):
        self.camera_manager.destroy()
        self.camera_manager = None

    def destroy(self):
        self.destroy_sensors()
        self.recorder.destroy()
        self.traffic.destroy()
        self.traffic_manager.shut_down()
        raise Exception("World has been destroyed!")

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        pass

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, client, world, recorder, hud, gamma_correction, args):
        self.sensor = None
        self.hud = hud
        self.bboxes = []
        self.image = []
        self.recorder = recorder
        self.args = args
 
        self.client = client
        self.world = world
        self.waypoints = world.get_map().generate_waypoints(1.0)

        print("Goodies loaded up!")

        self._camera_transforms = carla.Transform(carla.Location(x=-2.0, y=+0.0, z=20.0), carla.Rotation(pitch=8.0))

        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}, None],
            ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}, None],
        ]
        
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                bp.set_attribute('sensor_tick', str("0.05"))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            item[-1] = bp

    def create_sensors(self):
        batch = []
        SpawnActor = carla.command.SpawnActor
        for index in range(len(self.sensors)):
            batch.append(SpawnActor(self.sensors[index][-1],
                self._camera_transforms))

        responses = self.client.apply_batch_sync(batch, self.args.sync)
        for index in range(len(responses)):
            response = responses[index]
            if response.error:
                logging.error(response.error)
            else:
                self.sensors[index][3] = self.world.get_actor(response.actor_id)
        self.switch_waypoints()
        for index in range(len(responses)):
            response = responses[index]
            if response.error:
                logging.error(response.error)
            else:
                # We need to pass the lambda a weak reference to self to avoid
                # circular reference.
                weak_self = weakref.ref(self)
                
                if (index == 0):
                    self.sensors[index][3].listen(lambda image: CameraManager._parse_image(weak_self, image, 0))
                else:
                    self.sensors[index][3].listen(lambda image: CameraManager._parse_image(weak_self, image, 1))

    @staticmethod
    def _parse_image(weak_self, image, index):
        self = weak_self()
        if not self:
            return
        #print(self.world.get_actor(self.sensors[0][-2].id), self.sensors[0][-2].is_listening())
        #print(self.world.get_actor(self.sensors[1][-2].id), self.sensors[1][-2].is_listening())
        #print("Parsing {}!".format(index), len(self.bboxes), len(self.image))
        if self.sensors[index][0].startswith('sensor.camera.instance_segmentation'):
            if ((image.frame_number % 20) != 0):
                return
            
            image.convert(self.sensors[index][1])
            height = image.height
            width = image.width
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4)).copy()
            id_array = array.astype("uint16")
            semantic_array = id_array[:, :, 2]
            id_array = (array[:, :, 0] << 8) + array[:, :, 1]

            objs = {}
            objs["base_image"] = np.reshape(array, (image.height, image.width, 4)).copy()
            objs["id_array"] = id_array
            objs["semantic_array"] = semantic_array
            objs["height"] = image.height
            objs["width"] = image.width
            self.bboxes.append((objs, image.frame_number, image.timestamp))
        else:
            if ((image.frame_number % 20) != 0):
                return
            image.convert(self.sensors[index][1])
            height = image.height
            width = image.width
            fov = image.fov

            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (height, width, 4)).copy()
            self.image.append((array, image.frame_number, image.timestamp))
            
    def save_data(self):
        bboxes_indices = [self.bboxes[i][1:] for i in range(len(self.bboxes))]
        image_indices = [self.image[i][1:] for i in range(len(self.image))]
        print(bboxes_indices, image_indices)
        while (self.bboxes[0][1] < self.image[0][1]):
            self.bboxes.pop(0)
        if (len(self.bboxes) == 0):
            return
        while (self.image[0][1] < self.bboxes[0][1]):
            self.image.pop(0)
        if (len(self.image) == 0):
            return
        objs = self.bboxes.pop(0)[0]
        array = self.image.pop(0)[0]

        self.recorder.record_entry(array, objs)

    def switch_waypoints(self):
        selected_waypoint = random.choice(self.waypoints)
        self.waypoints = selected_waypoint.next(0.1)
        ApplyTransform = carla.command.ApplyTransform
        original_transform = selected_waypoint.transform
        location = original_transform.location
        rotation = original_transform.rotation
        transform = carla.Transform(carla.Location(location.x + random.uniform(-2.0, 2.0), location.y + random.uniform(-2.0, 2.0), location.z + 20 + random.uniform(-5.0, 5.0)), carla.Rotation(rotation.pitch + random.uniform(-5.0, 5.0), rotation.yaw + random.uniform(-5.0, 5.0), rotation.roll + random.uniform(-5.0, 5.0)))

        batch = []
        for index in range(len(self.sensors)):
            actor_id = self.sensors[index][3].id
            batch.append(ApplyTransform(actor_id, transform))
        
        self.client.apply_batch_sync(batch, False)

    def tick(self):
        self.switch_waypoints()
        if (len(self.bboxes) > 0) and (len(self.image) > 0):
            self.save_data()

    def destroy(self):
        for index in range(len(self.sensors)):
            if len(self.sensors[index]) >= 4:
                self.sensors[index][3].stop()
                self.sensors[index][3].destroy()
        

class Traffic(object):
    def __init__(self, client, traffic_manager, args):
        self.client = client
        self.args = args
        self.traffic_manager = traffic_manager
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.synchronous_master = False
        
    def get_actor_blueprints(self, world, filter, generation):
        bps = world.get_blueprint_library().filter(filter)

        if generation.lower() == "all":
            return bps

        # If the filter returns only one bp, we assume that this one needed
        # and therefore, we ignore the generation
        if len(bps) == 1:
            return bps

        try:
            int_generation = int(generation)
            # Check if generation is in available generations
            if int_generation in [1, 2, 3]:
                bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
                return bps
            else:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return []
        except:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []

    def instantiate_traffic(self):
        client = self.client
        args = self.args
        print("Grabbing world!")
        world = client.get_world()
        traffic_manager = self.traffic_manager

        print("Setting global settings.....")
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        settings = world.get_settings()
        if args.sync:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                self.synchronous_master = True
            else:
                self.synchronous_master = False
        else:
            print("You are currently in asynchronous mode, and traffic might experience some issues")

        blueprints = self.get_actor_blueprints(world, "vehicle.*", "All")
        if not blueprints:
            raise ValueError("Couldn't find any vehicles with the specified filters")
        blueprintsWalkers = self.get_actor_blueprints(world, "walker.pedestrian.*", "All")
        if not blueprintsWalkers:
            raise ValueError("Couldn't find any walkers with the specified filters")

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, self.synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                #Apply Offset in vertical to avoid collision spawning
                spawn_point.location.z += 2
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, self.synchronous_master)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, self.synchronous_master)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        self.all_actors = world.get_actors(self.all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if args.sync:
            world.tick()
        else:
            world.wait_for_tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(self.vehicles_list), len(self.walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)
   
    def destroy_traffic(self):
        if self.args.sync and self.synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        print('\ndestroying %d walkers' % len(self.walkers_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

        time.sleep(0.5)

    def destroy(self):
        self.destroy_traffic()

class Recorder(object):
    def __init__(self):
        self.coco_lock = Lock()
        self.coco_image_index = 0
        self.image_index = 0
        self.video_index = 0
        self.initialize_coco()
        shutil.rmtree("output/", ignore_errors=True)
        os.makedirs("output/", exist_ok=False)
        print("Started recorder")

    def add_image_entry(self, array, objs, base_image):
        for track_id in objs:
            bbox = objs[track_id]
            y_min = bbox["min_h"]
            y_max = bbox["max_h"]
            x_min = bbox["min_w"]
            x_max = bbox["max_w"]
            #cv2.line(array, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
            #cv2.line(array, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
            #cv2.line(array, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
            #cv2.line(array, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
        os.makedirs("output/%08d" % self.video_index, exist_ok=True)
        cv2.imwrite("output/%08d/%08d.png" % (self.video_index, self.image_index), array)
        cv2.imwrite("output/%08d/%08d_debug.png" % (self.video_index, self.image_index), base_image)

    def record_entry(self, array, segmentation):
        self.coco_lock.acquire()
        semantic_array = segmentation["semantic_array"]
        id_array = segmentation["id_array"]
        objs = {}

        desired_objs = [14,15,16]
        for h in range(segmentation["height"]):
            for w in range(segmentation["width"]):
                semantic_id = int(semantic_array[h][w])
                if semantic_id in desired_objs:
                    track_id = id_array[h][w]
                    if track_id not in objs:
                        objs[track_id] = {"min_w": w, "max_w": w, "min_h": h, "max_h": h}
                    if w < objs[track_id]["min_w"]:
                        objs[track_id]["min_w"] = w
                    elif w > objs[track_id]["max_w"]:
                        objs[track_id]["max_w"] = w
                    if h < objs[track_id]["min_h"]:
                        objs[track_id]["min_h"] = h
                    elif h > objs[track_id]["max_h"]:
                        objs[track_id]["max_h"] = h
        self.add_image_entry(array, objs, segmentation["base_image"])
        self.add_coco_entry(objs, segmentation["width"], segmentation["height"])
        self.image_index += 1
        self.coco_lock.release()

    def initialize_coco(self):
        self.coco_data = {}
        self.coco_data["info"] = {}
        self.coco_data["licenses"] = []
        self.coco_data["images"] = []
        self.coco_data["annotations"] = []
        self.coco_data["categories"] = [
            {"supercategory": "vehicle","id": 0,"name": "vehicle"}
        ]

    def add_coco_entry(self, objs, width, height):
        image_entry = {"video": int(self.video_index), "id": int(self.coco_image_index), "width": float(width), "height": float(height), "file_name": "%08d/%08d.png" % (self.video_index, self.image_index)}
        self.coco_data["images"].append(image_entry)
        for track_id in objs:
            bbox = objs[track_id]
            x = float(objs[track_id]["min_w"])
            y = float(objs[track_id]["min_h"])
            w = float(objs[track_id]["max_w"]) - x
            h = float(objs[track_id]["max_h"]) - y
            annotation_entry = {"id": int(track_id), "category_id": 0, "iscrowd": 0, "image_id": int(self.coco_image_index), "area": w*h, "bbox": [x, y, w, h]}
            self.coco_data["annotations"].append(annotation_entry)
        self.coco_image_index += 1

    def new_video(self):
        self.save_coco()
        self.coco_lock.acquire()
        self.video_index += 1
        self.image_index = 0
        self.coco_lock.release()

    def save_coco(self):
        self.coco_lock.acquire()
        with open("output/coco.json", "w+") as f:
            json.dump(self.coco_data, f, indent=4)
        self.coco_lock.release()

    def destroy(self):
        self.save_coco()
        print("Recorder stopped!")

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    pygame.init()
    world = None
    original_settings = None
    random.seed(1007)

    try:
        client = carla.Client(args.host, args.port)
        print("Client connected")
        client.set_timeout(2000.0)

        sim_world = client.get_world()

        if args.sync:
            original_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            if not settings.synchronous_mode:
                settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0/float(args.fps)

            sim_world.apply_settings(settings)
        else:
            settings = sim_world.get_settings()
            settings.fixed_delta_seconds = 1.0/float(args.fps)
            settings.synchronous_mode = False
            sim_world.apply_settings(settings)

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                  "experience some issues with the traffic simulation")

        hud = HUD(args.width, args.height)
        world = World(client, sim_world, hud, args)

        if args.sync:
            world.world.tick()
        else:
            world.world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            world.tick(clock)

    except Exception as e:
        print("EXCEPTION: ", e.message)
    finally:
        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description='CARLA Data Collection Client')
    argparser.add_argument(
        '-v', '--verbose', action='store_true', dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host', metavar='H', default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port', metavar='P', default=2000, type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot', action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res', metavar='WIDTHxHEIGHT', default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter', metavar='PATTERN', default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation', metavar='G', default='All',
        help='restrict to certain actor generation (values: "2","3","All" - default: "All")')
    argparser.add_argument(
        '--rolename', metavar='NAME', default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma', default=1.0, type=float,
        help='Gamma correction of the camera (default: 1.0)')
    argparser.add_argument(
        '--sync', action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '-n', '--number-of-vehicles', metavar='N', default=50, type=int,
        help='Number of vehicles (default: 50)')
    argparser.add_argument(
        '-w', '--number-of-walkers', metavar='W', default=20, type=int,
        help='Number of walkers (default: 20)')
    argparser.add_argument(
        '-t', '--total', metavar='T', default=86400, type=int,
        help='Total time elapsed')
    argparser.add_argument(
        '-f', '--fps', metavar='F', default=20, type=int,
        help='FPS')
    argparser.add_argument(
        '-s', '--switch', metavar='S', default=600, type=int,
        help='How many seconds per map and environment switch')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()
