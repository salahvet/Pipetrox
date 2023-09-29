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


#!/usr/bin/python
# -*- coding: utf-8 -*-


# iboss-2
# filename: __init__.py
# author: - Thomas Meschede
#
# modified:
#	- 2012 10 25 - Thomas Meschede

#__all__ = ["satellite","render"] 
#print(__path__)

import os
import sys
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#print(os.path.abspath("."))
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
#print(dir_path)

sys.path.insert(0,dir_path)

#__path__ += [__path__[0] + "/trimesh/trimesh"]

from . import mathhelp
from . import genutils
from . import generationgraph
