"""
piperator - plugin for generating pipes in blender
Copyright (C) 2019  Thomas Meschede

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


#!/usr/local/bin/python
# -*- coding: utf-8 -*-

# filename: generationgraph.py
# author: - Thomas Meschede
#
# usage: file loads ods file and puts data into a python array
#
# modified:
#	- 2019 9 21 - Thomas Meschede

"""script loads an ods file into python arrays"""

import mathhelp as mh
import time
import math
from math import cos, sin, pi, atan, sqrt
from os.path import dirname
import numpy as np
import itertools
import sys
import os
import logging
logger = logging.getLogger(__name__)

debugchain = []

def generate_pipe_wireframe(interfaces):
    #get list of interfaces and create a graph connecting them all
    #TODO: random.shuffle(interfaces)
    #TODO: remove doubles and generate a graph out of multipaths
    print(interfaces)
    logger.info(f"number of interfaces: {len(interfaces)}")
    pipechain=[]
    pipe_orig = interfaces[0]['loc'] #take first interface position as origin for pipe network object
    for i in range(len(interfaces)-1):
        loc,q = interfaces[i].values()
        loc2,q2 = interfaces[i+1].values()
        
        p1 = loc - pipe_orig
        p2 = p1 + q.rmat() @ mh.vec((0,0,1))
        p4 = loc2 - pipe_orig
        p3 = p4 + q2.rmat() @ mh.vec((0,0,1))
        
        # add two "helper" points 
        p0 = p1 + (p1-p2)
        p5 = p4 + (p4-p3)
        
        segment = [p0,p1,p2,p3,p4,p5]
        
        pipechain.extend(segment)
        debugchain.extend([[p0,p1],[p1,p2],[p2,p3],[p3,p4],[p4,p5]])
    return pipechain, pipe_orig

def generate_pipe_segment(q,p0,p1,p2,p3): 
            s0 = p1 - p0
            s1 = p2 - p1
            s2 = p3 - p2

            angle_start = mh.calc_angle_vec(s0,s1)
            angle_end = mh.calc_angle_vec(s1,s2)
            x_start = q.rmat() @ mh.vec((1.,.0,.0))
            
            eps = 1e-10
            if (abs(angle_end) < eps):
                x_end = x_start
                twist_end = 0.0
            else: 
                x_end = mh.normalized(mh.np.cross(s2,s1))
                twist_end = mh.calc_directed_angle(x_start, x_end, s1)
                       
            debugchain.append([p1,p1+x_start])
            debugchain.append([p2,p2+x_end])
            segment = dict(start = p1,
                             end = p2,
                             vec = s1,
                             orientation = q,
                             x_start = x_start,
                             x_end = x_end,
                             length = mh.norm(s1),
                             angle_end = angle_end*0.5,
                             angle_start = angle_start*0.5,
                             twist_end = twist_end)
                
            q1 = mh.quaternion.fromaxang(s1,twist_end)                
            q2 = mh.quaternion.fromaxang(x_end,-angle_end)
            q = q2 @ q1 @ q
            #q1 = mh.quaternion.fromaxang(mh.vec((1.,0.,0.0)),-angle_end)
            #q2 = mh.quaternion.fromaxang(mh.vec((0.,0.,1.0)),twist_end)               
            #q = q @ q1 @ q2
                             
            return segment, q    

def generate_pipe_description(pipechain):
    """generate a pipe as a list of pipe segments"""
    #generate a list of generic pipe segments
    segments = []
    #it doesn't matter which direction x-axis faces
    #therefore we can just take the "shortest" rotation
    q = mh.getquatrot((0.,0.,1.),pipechain[1]-pipechain[0]) #initialized q
    for i in range(0,len(pipechain)-3):
        seg,q = generate_pipe_segment(q,pipechain[i+0],
                                        pipechain[i+1],
                                        pipechain[i+2],
                                        pipechain[i+3])
        segments+=[seg]
    
    return segments

if __name__ == '__main__':
    import doctest
    doctest.testmod()
#    doctest.run_docstring_examples(rotverts, globals())
