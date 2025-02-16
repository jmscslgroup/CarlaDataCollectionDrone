from multiprocessing import Lock
import shutil
import os
import cv2
import json

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
