#!/bin/bash

module load singularity

singularity exec --nv /home/jphillips/images/csci4850-2022-Fall.sif ./gan.py