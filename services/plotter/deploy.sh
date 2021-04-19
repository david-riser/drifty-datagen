#!/usr/bin/env bash

docker run -p 8888:8888 \
       -v ${PWD}/../generator/data:/data \
       plotter_image
