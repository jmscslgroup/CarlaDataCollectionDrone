import random
import numpy as np
import weakref
import logging

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, carla, client, world, output_queue, gamma_correction, args):
        self.carla = carla
        self.sensor = None
        self.bboxes = []
        self.image = []
        self.output_queue = output_queue
        self.args = args
 
        self.client = client
        self.world = world
        self.waypoints = world.get_map().generate_waypoints(1.0)

        print("Goodies loaded up!")

        self._camera_transforms = self.carla.Transform(self.carla.Location(x=-2.0, y=+0.0, z=20.0), self.carla.Rotation(pitch=8.0))

        self.sensors = [
            ['sensor.camera.rgb', self.carla.ColorConverter.Raw, 'Camera RGB', {}, None],
            ['sensor.camera.instance_segmentation', self.carla.ColorConverter.Raw, 'Camera Instance Segmentation (Raw)', {}, None],
        ]
        
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(self.args.width))
                bp.set_attribute('image_size_y', str(self.args.height))
                bp.set_attribute('sensor_tick', str("0.05"))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            item[-1] = bp

    def create_sensors(self):
        batch = []
        SpawnActor = self.carla.command.SpawnActor
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

        self.output_queue.put((array, objs))

    def switch_waypoints(self):
        selected_waypoint = random.choice(self.waypoints)
        self.waypoints = selected_waypoint.next(0.1)
        ApplyTransform = self.carla.command.ApplyTransform
        original_transform = selected_waypoint.transform
        location = original_transform.location
        rotation = original_transform.rotation
        transform = self.carla.Transform(self.carla.Location(location.x + random.uniform(-2.0, 2.0), location.y + random.uniform(-2.0, 2.0), location.z + 20 + random.uniform(-5.0, 5.0)), self.carla.Rotation(rotation.pitch + random.uniform(-5.0, 5.0), rotation.yaw + random.uniform(-5.0, 5.0), rotation.roll + random.uniform(-5.0, 5.0)))

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