#!/usr/bin/env python

import argparse
import logging
import random
from multiprocessing import Queue, Process, Value, Lock
from Manager import Manager
# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args):
    world = None
    original_settings = None
    random.seed(1007)

    try:
        manager = Manager(args)
        manager.run_episodes()
    except Exception as e:
        print("EXCEPTION: ", e.message)
    finally:
        if world is not None:
            world.destroy()


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
        '--number-of-simulators', metavar='SIMS', default=2, type=int,
        help='Number of Simulators (default: 2)')
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
