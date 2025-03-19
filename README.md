# Introduction 
This serves as a proof-of-concept demonstration of how to run CARLA in parallel and at scale indefinitely while avoiding the numerous stability issues extant for the last several years. You have to run all simulation runs with a CARLA server that has a lifetime the same as the episode, and same for the Python Client processes involved.

This proof-of-concept creates a 24 hour "drone" traffic dataset in the COCO format. The default total dataset size is 24 hours, with each episode being 10 minutes long.

# How To Use
## Dependencies
1. CARLA 0.10.0 (Unreal Engine 5 version)
2. pip - opencv-python, importlib, and the CARLA python library.

Execute `python3 data_collection.py`. The dataset will begin compiling immediately.