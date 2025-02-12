FROM carlasim/carla:0.9.15
USER root
RUN apt-get update && apt-get install -y xdg-user-dirs xdg-utils alsa-utils && apt-get clean
USER carla
