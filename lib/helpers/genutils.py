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

# iboss-2
# filename: odspy.py
# author: - Thomas Meschede
#
# usage: file loads ods file and puts data into a python array
#
# modified:
#	- 2012 10 25 - Thomas Meschede

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
inblender = False
try:
    import bpy
    import bpy.props
    from bpy.props import FloatVectorProperty
    import bmesh
    inblender = True

    import mathutils
    V3D = mathutils.Vector
except BaseException:
    logging.info("genutils: not in blender")


# curfp=bpy.data.filepath
# print(os.getcwd())
# print(dirname(__file__))
# sys.path.append(dirname(__file__)) #add current directory to blender path
# print(np.array(sys.path))

pi2 = pi * 2.0
pi05 = pi * 0.5


def vec(x): return np.array(x)


def vec3(x, y, z): return np.array((x, y, z))


def MoI(x, y, z): return np.array(((x, 0, 0), (0, y, 0), (0, 0, z)))


def norm(x): return np.sqrt(x.dot(x))


def normalized(x): return x * (1.0 / norm(x))


#def normalized(x): return x / np.sqrt(np.sum(x * x))


def copyobject(oldobjname):
    me = bpy.data.objects[oldobjname].data

    ob = bpy.data.objects.new(oldobjname + "_cp", me)  # create a new object
    ob.data = me          # link the mesh data to the object
    scene = bpy.context.scene           # get the current scene
    # link the object into the scene
    scene.objects.link(ob)
    return ob


def assign_material(me, mat):
    # Assign material to mesh
    if me.materials:
        # assign to 1st material slot
        me.materials[0] = mat
    else:
        # no slots
        me.materials.append(mat)
    return me


def calc_polygon_center(polygon, vertices):
    center = vec(0.0, 0.0, 0.0)
    for i_v in polygon.vertices:
        center += vertices[i_v]
    return center / len(vertices)


def listify(numpyarray):
    if isinstance(numpyarray, list) or isinstance(numpyarray, np.ndarray):
        return [listify(subarray) for subarray in numpyarray]
    else:
        return numpyarray


def flatten(verts):
    # return np.array(np.ravel(verts).reshape(-1,3))
    # return np.array(np.ravel(verts).reshape(-1,3))
    # return np.hstack(verts.flat)
    return list(itertools.chain.from_iterable(verts))

def genempty(loc, rot_quat):
    o = bpy.data.objects.new("empty", None ) 
    o.empty_display_type='ARROWS'
    o.location = loc
    o.rotation_mode = 'QUATERNION'
    o.rotation_quaternion = rot_quat
    bpy.context.scene.collection.objects.link(o)
    return o

def genmeshfrompydata(verts, edges=None, faces=None, meshname="genfrompydata"):
    #print(verts, edges, faces)
    """
    faces and edges have to be python lists!!  no numpy arrays!!
    """
    #verts = listify(verts)
    #import code
    # code.interact(local=locals())
    me = bpy.data.meshes.new(meshname)  # create a new mesh
    me.from_pydata(verts, edges, faces)
    me.update()      # update the mesh with the new data
    return me


def genobjfrompydata(verts, edges=[], faces=[], objname="genfrompydata"):
    me = genmeshfrompydata(verts, edges, faces)  # create a new mesh
    return genobjfrommesh(objname, me)


def genmesh(meshname, verts=[], mat=None):
    edges = []
    verts, edges, faces = creategeometry(verts)
    me = genmeshfrompydata(verts, edges, faces)  # create a new mesh

    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
    bm.to_mesh(me)
    me.update()
    bm.clear()
    bm.free()

    assign_material(me, mat)
    return me


def genobject(objname, verts=[], mat = None, 
              use_smooth = True):
    me = genmesh(objname, verts, mat)
    ob = bpy.data.objects.new(objname, me)  # create a new object
    for f in me.polygons:
        f.use_smooth = use_smooth
    ob.data = me          # link the mesh data to the object
    #scene = bpy.context.scene           # get the current scene
    # link the object into the scene
    #scene.collection.objects.link(ob)
    return ob

# group hints:
# https://blender.stackexchange.com/questions/30737/add-a-group-instance-with-python


def genobjfromgroup(objname, groupname):
    group = bpy.data.groups[groupname]
    ob = bpy.data.objects.new(objname, None)  # create a new object
    ob.dupli_type = 'GROUP'
    ob.dupli_group = group  # link the mesh data to the object
    scene = bpy.context.scene   # get the current scene
    scene.objects.link(ob)      # link the object into the scene
    return ob

# generate object from pre-existing blender mesh


def genobjfrommesh(objname, me):
    ob = bpy.data.objects.new(objname, me)  # create a new object
    ob.data = me          # link the mesh data to the object
    scene = bpy.context.scene           # get the current scene
    # link the object into the scene
    scene.collection.objects.link(ob)
    return ob


def genobjandremovedoubles(verts, mat=None, name="genutils"):
    obj = genobject(name, verts, mat=mat)

    selectsingleobj(obj)
    # now superseded by bmesh in genmesh
    # bpy.ops.object.editmode_toggle()
    # bpy.ops.mesh.select_all(action='SELECT')
    # bpy.ops.mesh.remove_doubles()#threshhold=0.01)
    # bpy.ops.object.editmode_toggle()
    return obj


def gen3dlist(size, initializer=None):
    vm = [None] * size[0]
    for i in range(size[0]):
        vm[i] = [None] * size[1]
        for j in range(size[1]):
            vm[i][j] = [initializer] * size[2]
    return vm


def getactiveobj():
    return bpy.context.scene.objects.active


def selectsingleobj(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select = True
    bpy.context.scene.objects.active = obj


def smoothshading(obj):
    # selectsingleobj(obj)
    # bpy.ops.object.shade_smooth()

    for poly in obj.data.polygons:
        poly.use_smooth = True


def joinobjects(objlist):
    obj = genobject("joined")
    selectobj(obj)
    for i in objlist:
        i.select = True
    bpy.ops.object.join()
    return obj


def rotverts(verts, quat):
    """
    >>> verts = createcube(1.0)
    >>> len(np.ravel(verts).reshape(-1,3).tolist())
    24
    >>> rotated_verts = rotverts(verts,mh.quaternion((0.1,0.2,0.3,0.4)))
    >>> len(np.ravel(rotated_verts).reshape(-1,3).tolist())
    24
    >>> rotated_verts = rotverts(verts,mh.getquatrot((0,0,1),(0.1,0.2,0.3)))
    >>> len(creategeometry(rotated_verts)[0])
    24
    """
    rot = quat.rmat()
    newverts = []
    for f in verts:
        tmp = []
        for v in f:
            tmp.append(rot@v)
        newverts.append(vec(tmp))
    return newverts


def translateverts(verts, translate):
    """

    >>> verts = createcylinder(r=1.0,b1=0.0,b2=1.0,res=3, closed=(1,1))
    >>> translated_verts = translateverts(verts, (0.1,0.1,0.1))

    >>> verts = createpipesegment(innerradius = 0.09,length = 1.0)
    >>> translated_verts = translateverts(verts, (0.1,0.1,0.1))

    """
    newverts = []
    #verts = vec(verts)
    #translate = vec(translate)
    for face in verts:
        tmp = []
        for v in face:
            #import code
            # code.interact(local=locals())
            tmp.append(vec(v) + vec(translate))
        newverts.append(vec(tmp))
    return newverts


def translatebody(vlist, translate):
    tmp = np.array(vlist) + np.array(translate)
    return tmp.tolist()


def creategeometry(verts):
    faces = []
    edges = []
    faceoffset = 0
    for ver in verts:
        if len(ver) == 4:
            faces.append(
                (faceoffset + 0,
                 faceoffset + 1,
                 faceoffset + 2,
                 faceoffset + 3))
            faceoffset += 4
        elif len(ver) == 3:
            faces.append((faceoffset + 0, faceoffset + 1, faceoffset + 2))
            faceoffset += 3
        elif len(ver) == 2:
            edges.append((faceoffset + 0, faceoffset + 1))
            faceoffset += 2
    return flatten(verts), edges, faces


def createbox(halfbounds):
    verts = []

    vb = mh.get_boundaryvertices(halfbounds)

    faces = ((2, 0, 4, 6), (3, 7, 5, 1), (0, 1, 5, 4),
             (2, 6, 7, 3), (0, 2, 3, 1), (4, 5, 7, 6))

    verts = [(vb[a], vb[b], vb[c], vb[d]) for a, b, c, d in faces]

    return verts


def createquadverts(size=(1, 1), pos=(0, 0, 0), rot=(0, 0, 0)):
    newverts = [(0, 0, 0), (size[0], 0, 0),
                (size[0], size[1], 0), (0, size[1], 0)]
    rot = Euler(rot).to_matrix()
    verts = []
    for i in newverts:
        verts.append(rot * Vector(i) + Vector(pos))
    return [verts]


def createcube(edgelength):
    verts = []
    r = edgelength / sqrt(2)
    e = edgelength * 0.5
    u, o = -e, e
    a = pi / 4.0
    for i in range(4):
        a += pi * 0.5
        x, y = r * sin(a), r * cos(a)
        b = a + pi * 0.5
        x2, y2 = r * sin(b), r * cos(b)
        verts.append([(x, y, u), (x2, y2, u), (x2, y2, o), (x, y, o)])
    verts.append([(e, e, u), (-e, e, u), (-e, -e, u), (e, -e, u)])
    verts.append([(e, -e, o), (-e, -e, o), (-e, e, o), (e, e, o)])

    return verts


def createpipesegment(innerradius,
                      length,
                      ends_closed=(0, 0),
                      interface_angle1=0.0,
                      interface_angle2=0.0,
                      outerradius=0.0,
                      angle2_twist=0.0,
                      res=8):
    """
    create a pipe segment

    >>> cy = createpipesegment(innerradius = 0.09,length = 1.0)
    >>> cy1 = translateverts(cy,(0.1,0.2,0.3))
    >>> cy2 = rotverts(cy,mh.getquatrot((0,0,1),(0.1,0.2,0.3)))
    >>> len(cy2)
    8
    >>> g = creategeometry(cy2)
    
    """
    verts = []
    r = innerradius
    r05 = r * 0.5
    for i in range(res):
        a = i * 2 * pi / res
        x, y = r * sin(a), r * cos(a)
        b1_1 = math.tan(interface_angle1) * y
        b2_1 = length - math.tan(interface_angle2) * r * cos(a + angle2_twist)
        a = (i + 1) * 2 * pi / res
        x2, y2 = r * sin(a), r * cos(a)
        #rotate x2 and y2 according to twist angle
        b1_2 = math.tan(interface_angle1) * y2
        b2_2 = length - math.tan(interface_angle2) * r * cos(a + angle2_twist)
        verts.append(
            vec([(x, y, b1_1), 
                 (x2, y2, b1_2), 
                 (x2, y2, b2_2), 
                 (x, y, b2_1)]))
        # caps
        if ends_closed[0]:
            verts.append(vec([(x, y, b1_1), (x2, y2, b1_2), (.0, .0, .0)]))
        if ends_closed[1]:
            verts.append(vec([(x, y, b2_1), (x2, y2, b2_2), (.0, .0, length)]))
    return verts


def createcylinder(r, b1, b2, res, closed=(1, 1)):
    """
    >>> len(createcylinder(r=1.0,b1=0.0,b2=1.0,res=3, closed=(0,0)))
    3

    """
    verts = []
    for i in range(res):
        a = i * 2 * pi / res
        x, y = r * sin(a), r * cos(a)
        a = (i + 1) * 2 * pi / res
        x2, y2 = r * sin(a), r * cos(a)
        verts.append(
            np.array([(x, y, b1), (x2, y2, b1), (x2, y2, b2), (x, y, b2)]))
        # caps
        if closed[0]:
            verts.append(np.array([(x, y, b1), (x2, y2, b1), (0, 0, b1)]))
        if closed[1]:
            verts.append(np.array([(x, y, b2), (x2, y2, b2), (0, 0, b2)]))
    return verts


def createcone(r, h, res):
    verts = []
    for i in range(res):
        a = i * 2 * pi / res
        x, y = r * sin(a), r * cos(a)
        a = (i + 1) * 2 * pi / res
        x2, y2 = r * sin(a), r * cos(a)
        verts.append([(0.0, 0.0, 0.0), (x, y, -h), (x2, y2, -h)])
    return verts


def arrow(pos, vec, width=0.1, mat=None):
    # Generate the tracking quaternion that will account for the rotation
    vec = -V3D(vec)
    quat = vec.to_track_quat("Z", "X")
    length = vec.length

    # generate arrow-obj
    r = width * 0.5
    res = 8
    verts = createcylinder(r, 0.0 + 3 * r, length, res)
    verts.extend(createcone(r * 2.5, -r * 4, res))

    # print(np.array(verts))

    ar = genobjandremovedoubles(verts, mat)

    # obj2=cpobj("pfeil.schaft")
    ar.location = pos
    # obj2.scale=(width,width,length-lengthdiv)
    ar.rotation_mode = "QUATERNION"
    ar.rotation_quaternion = quat
    return ar


def joinblenderobjs(objs):
    import bpy
    scene = bpy.context.scene

    ctx = bpy.context.copy()

    # one of the objects to join
    ctx['active_object'] = objs[0]
    ctx['selected_objects'] = objs

    ctx['selected_editable_bases'] = [scene.object_bases[ob.name]
                                      for ob in objs]
    bpy.ops.object.join(ctx)


def move_along_local_axis(obj, local_vector):
    # one blender unit in x-direction
    vec = mathutils.Vector(local_vector)
    inv = obj.matrix_world.copy()
    inv.invert()
    # vec aligned to local axis
    vec_rot = vec * inv
    obj.location = obj.location + vec_rot
    return obj


def axis3d(mat, size=1.0):
    width = 0.01
    a1 = arrow((1, 0, 0), (1.0, 0.0, 0.0), width, mat)
    a2 = arrow((0, 1, 0), (0.0, 1.0, 0.0), width, mat)
    a3 = arrow((0, 0, 1), (0.0, 0.0, 1.0), width, mat)
    joinblenderobjs([a1, a2, a3])
    selectsingleobj(a1)
    bpy.context.scene.cursor_location = (0.0, 0.0, 0.0)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    return a1

# return w,x,y,z


def getquatrot(u, v):
    k_cos_theta = np.dot(u, v)
    u2 = np.sum(u * u)
    v2 = np.sum(v * v)
    k = sqrt(u2 * v2)

    if (k_cos_theta / k == -1):
        # 180 degree rotation around any orthogonal vector
        x, y, z = u / np.sqrt(u2)
        return 0, x, y, z

    w = k_cos_theta + k
    x, y, z = np.cross(u, v)
    return normalized(np.array((w, x, y, z)))


def vertgroups2vertsandfaces(vertgroups, epsilon=0.00001):
    tris = []
    quads = []
    verts = []
    vertnumbers = {}
    for vertg in vertgroups:
        face = []
        for v_ in vertg:
            v = tuple(v_)
            vnum = vertnumbers.get(v, None)
            if vnum is None:
                vnum = len(verts)
                vertnumbers[v] = vnum
                verts.append(v)

            face.append(vnum)

        if len(face) < 4:
            tris.append(face)
        else:
            quads.append(face)

    return np.array(verts).reshape(-1, 3), np.array(tris).reshape(-1,
                                                                  3), np.array(quads).reshape(-1, 4)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
#    doctest.run_docstring_examples(rotverts, globals())

#box = createbox((1.0,2.0,3.0))
#v,t,q = vertgroups2vertsandfaces(box)


#verts = createcylinder(b1=-0.01,b2=0.01,r=0.1,res=10)
#intfmesh = genmesh("interface",verts)
#intf = genobjfrommesh("interface",intfmesh)

#bpy.ops.mesh.primitive_cube_add(radius=1, view_align=False, enter_editmode=False, location=(0, 0, 0), layers=(True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))


# targetpos=np.array([1.0,-.3,.9])
#ex = np.array((1.0,0.0,0.0))
#ey = np.array((0.0,1.0,0.0))
#ez = np.array((0.0,0.0,1.0))
#to = arrow(targetpos,(0.3,0.0,0.0),0.1, mat)

# getquatrot(-ez,targetpos)
#w,x,y,z = getquatrot(-ez,targetpos)
# print(w,x,y,z)
#quat = mathutils.Quaternion([w,x,y,z])

#po = arrow((0,0,0),(1.0,0.0,0.0),0.1, mat)
# po.rotation_mode="QUATERNION"
#po.rotation_quaternion = quat


#mat = bpy.data.materials.new(name="white")
# mat.diffuse_color=(1.0,1.0,1.0)
# mat.use_shadeless=True
# axis3d(mat)


# blender -b seher.blend -P rendersat.py render
