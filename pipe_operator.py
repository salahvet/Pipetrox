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

#Builtin Modules:     bpy, bpy.data, bpy.ops, bpy.props, bpy.types, bpy.context, bpy.utils, bgl, blf, mathutils
#Convenience Imports: from mathutils import *; from math import *
#Convenience Variables: C = bpy.context, D = bpy.data

__author__ = "yeus <Thomas.Meschede@web.de>"
__status__ = "test"
__version__ = "0.09"
__date__ = "2019 Oct 23rd"

import sys, os
import bpy
import bmesh
import random
import numpy as np
import math
import collections
from bpy_extras.object_utils import AddObjectHelper
import mathutils as mu
import networkx as nx

import logging
logger = logging.getLogger(__name__)

def printinfo(msg):
    logger.info(msg)
    print(msg)
    return msg

import importlib
import helpers.mathhelp as mh
import helpers.genutils as gu
import helpers.generationgraph as gg
importlib.reload(mh)
importlib.reload(gu)
importlib.reload(gg)

def get_slice_edge(ob, plane_normal, plane_loc, offset = 0.1):
    me = ob.data
    
    bm = bmesh.new()   # create an empty BMesh
    bm.from_mesh(me)   # fill it in from a Mesh

    # read more about the slicing operation here:
    # https://blender.stackexchange.com/questions/3619/how-to-cut-a-mesh-into-smaller-pieces-with-python/3623
    # https://blender.stackexchange.com/questions/90724/what-is-the-best-way-to-copy-append-geometry-from-one-bmesh-to-another
    ret = bmesh.ops.bisect_plane(bm,
                geom=bm.verts[:]+bm.edges[:]+bm.faces[:], 
                plane_co=plane_loc, plane_no=plane_normal,
                clear_outer=False) 
    
    vs = [v for v in ret['geom_cut'] if isinstance(v, bmesh.types.BMVert)]
    es = [e for e in ret['geom_cut'] if isinstance(e, bmesh.types.BMEdge)]
    
    #print(vs[0].normal)
    
    vertmap = {}
    vs_cut = []
    vs_cut_off = []
    for i,v in enumerate(vs):
        newvec = v.co+v.normal*offset
        vs_cut.append(v.co)
        vs_cut_off.append(newvec)
        vertmap[v.index] = i

    es_cut = []
    for e in es:
        idx0, idx1 = e.verts[0].index, e.verts[1].index
        es_cut.append([vertmap[idx0], vertmap[idx1]])

        
    #bm.to_mesh(me)
    bm.free()  # free and prevent further access

    return vs_cut, vs_cut_off, es_cut

def get_interfaces():
    interfaces = []
    for o in bpy.context.selected_objects:
        m = o.data
        n = m.polygons[0].normal
        q = o.matrix_world.to_quaternion()
        n = q @ n #get normal vector in world coordinates
        #q2 = mh.getquatrot((0,0,1),n) #TODO: if there is now rotation available
        interfaces.append({'loc' : mh.vec(o.location),
                           'orientation' : mh.quaternion(q)})   #interface is defined through location and orientation
    return interfaces

class pathfinder:
    def __init__(self):
        self.occupied_verts = collections.defaultdict(int)
        self.occupied_edges = collections.defaultdict(int)
    
    def generate_pathfinding_mesh(self,sel_obj, offset,
                                    layers = 2):
        if layers != 2: raise NotImplementedError
        #TODO: more or less than 2 layers
        me = sel_obj.data
        
        bm = bmesh.new()   # create an empty BMesh
        bm.from_mesh(me)   # fill it in from a Mesh
        
        bm_path = bmesh.new()

        #geom = bm.verts[:]+bm.edges[:]+bm.faces[:]
        #bmesh.ops.duplicate(bm, 
        #                    geom = geom,
        #                    dest = bm_path)
        #bm_path.verts = bm.verts.copy_from(bm.verts)
        #bmesh.ops.wireframe()
        #bmesh.ops.convex_hull()
        
        layer1verts = []
        vertmap = {}
        layeredges = []
        layer2verts = []
        vertmap2 = {}
        i = 0
        for v in bm.verts:
            #make sure v.normal is normalized
            #TODO: use custom vertex class with its own properties
            v1 = v.co + v.normal * offset[0]
            layer1verts.append(v1)
            v2 = v.co + v.normal * (offset[0] + offset[1])
            layer2verts.append(v2)
            vertmap[v.index] = i
            i+=1
        
        for e in bm.edges:
            idx0, idx1 = e.verts[0].index, e.verts[1].index
            layeredges.append([vertmap[idx0], vertmap[idx1]])
            
        #merge the two separate meshes
        maxvert = max(vertmap.values())+1
        verts = layer1verts + layer2verts
        layer2edges = [(e1+maxvert, e2+maxvert) for e1,e2 in layeredges]
        conn_edges = [(v, v+maxvert) for v in vertmap.values()]
        edges = layeredges + layer2edges + conn_edges
        
        
        bm.free()  # free and prevent further access
        
        #merge vertices that are closer than a certain value to avoid collisions
        #bmesh.ops.find_doubles
        #bmesh.ops.remove_doubles
        #bmesh.ops.automerge #https://docs.blender.org/api/2.80/bmesh.ops.html#bmesh.ops.automerge
        # an alternativ would be to mark all vertices that are closer than 
        #a certain value as "occupied"
        self.verts = verts
        self.edges = edges
        return verts, edges

    def generate_pathfinding_mesh_from_slices(self,sel_object, interfaces, offset):
        p_start, p_end = interfaces[0]['loc'],interfaces[-1]['loc']
        p_start_n = interfaces[0]['orientation'].rmat() @ mh.vec3(0.,0.,1.)
        p_end_n = interfaces[-1]['orientation'].rmat() @ mh.vec3(0.,0.,1.)
        
        # calculate a plane connecting the two interfaces
        # this plane will get used for the pipe as a constraint
        # the pipe will get offset in the plane as well:
        # the plane ets defined by three points: interface1, interface2
        # and the interface z-axis from interface1
        z_start = interfaces[0]['orientation'].rmat() @ mh.vec((0.,0.,1.))
        plane_normal = np.cross(z_start, p_end-p_start)
        #print(plane_normal)
        
        #create edge points around the mesh
        vs_cut, vs_cut_off, es_cut = get_slice_edge(sel_object, plane_normal, p_start, offset)
        
        self.verts = vs_cut_off
        self.edges = es_cut
        return vs_cut_off, es_cut
    
    def addgeometry(self, verts, edge_numbers):
        #find two closest points to entry and exit points into the network    
        size = len(self.verts)
        kd = mu.kdtree.KDTree(size)
        for i, v in enumerate(self.verts):
            kd.insert(v, i)
        kd.balance()

        #TODO: only find neighbouring verts that are "unoccupied"
        #TODO: if start is occupied fin more start-choices
        #(co_start, idx_start), (co_start2, idx_start2) = kd.find_n(p_start,2)
        e_s = kd.find_n(verts[0],edge_numbers[0]) #starting edges
        #co_end, idx_end, _ = kd.find_n(p_end,2)
        e_e = kd.find_n(verts[1],edge_numbers[1]) #ending edges
        
        #import ipdb; ipdb.set_trace()
        start_occ = [self.occupied_verts[idx] for v,idx,d in e_s]
        start_occupied = not(0 in start_occ)
        end_occ = [self.occupied_verts[idx] for v,idx,d in e_e]
        end_occupied = not(0 in end_occ)
        
        if start_occupied or end_occupied:
            return (start_occupied,  end_occupied), "occupied"
        
        offset_idx = size-1
        #print(f"\nvertex#: {len(vs_cut)}")
        #print(f"and max_idx: {max_idx}")
        
        #create starting/ending edges
        #             e_s[0]
        #            /
        # new_vertex-e_s[1]
        #            \
        #             e_s[X]
        #
        
        idx = [offset_idx+1, offset_idx+2]

        self.verts.append(mu.Vector(verts[0]))
        self.verts.append(mu.Vector(verts[1]))
        self.edges += [[idx[0],v[1]] for v in e_s]
        self.edges += [[idx[1],v[1]] for v in e_e]
        
        return idx, "success"#self.verts, self.edges
    
    def get_free_mesh(self):
        #import ipdb; ipdb.set_trace()
        verts = []
        vertmap = {}
        counter = 0
        for i,v in enumerate(self.verts):
            if self.occupied_verts[i] == 0:
                verts.append(v)
                vertmap[i] = counter
                counter+=1
                
        edges = []
        for e in self.edges:
            if (self.occupied_verts[e[0]] == 0) and (self.occupied_verts[e[1]] == 0): 
                edges.append((vertmap[e[0]],vertmap[e[1]]))
        return verts, edges, vertmap
    
    def increase_occupation(self,vertex_idxs, vertmap):
        #map occupation back to original pathmap
        vertmap = {v: k for k, v in vertmap.items()}
        for i in vertex_idxs: self.occupied_verts[vertmap[i]] += 1

class object_catalog:
    def __init__(self):
        self.objs = {}
        
    def append_if_not_exist(self, key, obj):
        if key in self.objs:
            return self.objs[key]
        
        self.objs[key] = obj
        
        return obj
    
def get_interfaces_from_object_faces(sel_object):
    interfaces=[]
    o = sel_object
    #for o in bpy.context.selected_objects:
    fs = list(o.data.polygons)
    for f in fs:
        p = f.center
        n = f.normal
        q_obj = o.matrix_world.to_quaternion()
        n = q_obj @ n #get normal vector in world coordinates
        q_pipe = mh.getquatrot((0.,0.,1.),n)
        #q2 = mh.getquatrot((0,0,1),n) #TODO: if there is now rotation available
        interfaces.append({'loc' : mh.vec(p),
                            'orientation' : q_pipe})   #interface is defined through location and orientation
    return interfaces

def create_wires_around_mesh(sel_object, interfaces, offset = 0.1,
                                    pathmesh = None,
                                    surfaceglue = True):                                    
    #print(interfaces)
    #import ipdb; ipdb.set_trace()
    p_start, p_end = interfaces[0]['loc'],interfaces[-1]['loc']
    p_start_n = interfaces[0]['orientation'].rmat() @ mh.vec3(0.,0.,1.)
    p_end_n = interfaces[-1]['orientation'].rmat() @ mh.vec3(0.,0.,1.)
    
    #build first three points of chain
    # ps2 and pe0 are the entry & exit points into the slicing-edge
    ps0 = p_start - (p_start_n * offset)
    ps1 = p_start
    ps2 = p_start + (p_start_n * offset)            
    pe0 = p_end + (p_end_n * offset)
    pe1 = p_end
    pe2 = p_end - (p_end_n * offset)

    #create starting/ending edges
    #             e_s[0]
    #            /
    # ps0-ps1-ps2-e_s[1]
    #            \
    #             e_s[X]
    #
    #breakpoint()
    #import ipdb; ipdb.set_trace()
    new_verts = (mu.Vector(ps2),mu.Vector(pe0))
    #TODO: adapt edge number argument to number of face vertices
    verts_idx, state = pathmesh.addgeometry(new_verts, [4,4])
    if state == "occupied":
        return verts_idx, "occupied" #return which interfaces are occupied
    ps2_idx, pe0_idx = verts_idx
    verts, edges, vertmap = pathmesh.get_free_mesh()
    ps2_idx, pe0_idx = vertmap[ps2_idx], vertmap[pe0_idx] #map to new free mesh
    #verts, edges = pathmesh.verts, pathmesh.edges

    #add egde chain of the slice
    loop = []
    #print(edges)
    #TODO: do this balancing in the pathfinder algorithm
    for i1,i2 in edges:
        length = (verts[i1] - verts[i2]).length
        # weight can be used instad of "length": this way
        # we can make the algorithm go around occupied vertices
        # or influence it to prefer vertices with a certain
        # vertex paint in blender etc...
        # TODO: make vertex paint influence the weight
        # TODO: influence weight by "layer height"
        #       - this means "lower" layers should be preferred for example
        depsgraph = bpy.context.evaluated_depsgraph_get()
        _,location,_,_ = sel_object.closest_point_on_mesh(verts[i1])
        distance = (verts[i1] - location).length
        if surfaceglue: weight = length+distance*2.0
        else: weight = length
        loop.append([i1,i2,{"length":length,
                            "weight":weight}])
    
    #print(loop)
    g = nx.Graph()
    g.add_edges_from(loop)
    path = nx.shortest_path(g,
                    source=ps2_idx,
                    target=pe0_idx,
                    weight='weight')
    
    pathmesh.increase_occupation(path, vertmap)
                    
    # build connected edges out of graph
    pathvertices = [mh.vec(verts[i]) for i in path]
                                
    #wire_verts = [ps0,ps1,ps2]+pathvertices+[pe0,pe1,pe2]
    #wire_verts = [ps0,ps1,ps2]+pathvertices[1:-1]+[pe0,pe1,pe2]    
    wire_verts = [ps0,ps1]+pathvertices+[pe1,pe2]    
        
    return wire_verts, "success"

def render_pipes(pipechain, radius, debug = False): 
    pipe_orig = mh.vec3(0,0,0)#pipechain[1]
    pipesegments = gg.generate_pipe_description(pipechain)
    
    #convert to edges
    if debug >= 1:
        edges = []
        for i in range(len(pipechain)-1):
            edges += [[pipechain[i], pipechain[i+1]]]
        gg.debugchain += edges
    

    #render pipesegments
    vs=[]
    for s in pipesegments:
        cy = gu.createpipesegment(innerradius = radius,length = s["length"],
                                  interface_angle1 = s["angle_start"],
                                  interface_angle2 = s["angle_end"],
                                  angle2_twist = s["twist_end"],
                                  res=20)
                                  
        #get twist angle for the pipe interface 
        #twist = 
        #get orientation so that axis is aligned
        if debug >= 2: #add empty for debugging
            o = bpy.data.objects.new( "empty", None ) 
            o.empty_display_type='ARROWS'
            o.location = s["start"] + pipe_orig
            o.rotation_mode = 'QUATERNION'
            o.rotation_quaternion = s["orientation"]
            bpy.context.scene.collection.objects.link(o)
        cy = gu.rotverts(cy,s["orientation"])
        cy = gu.translateverts(cy,s["start"])
        vs += cy

    o2 = gu.genobject("pipe",vs)
    #o2.rotation_mode = 'QUATERNION'
    #o2.rotation_quaternion = q
    o2.location = pipe_orig#calc_polygon_center
    
    #create edges
    if debug >= 1:
        o_edge = gu.genobject("pipe_edges",gg.debugchain)
        o_edge.location = pipe_orig
        
    return o2

def render_components(pipechain, radius, mode, randomnes, obj_catalog):
    threshold = randomnes[0]
    if threshold == 0:
        return []
    
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),"./objects/pipe_objects.stl")
    me = bpy.ops.import_mesh.stl(filepath = filepath,
                                global_scale=radius*1.5)    
    if me != {'FINISHED'}:
        return rintinfo("not able to import STL files!")
    obj = bpy.context.selected_objects[0]
        
    obj = obj_catalog.append_if_not_exist("pipe_flange", obj)
    
    objs = []
    pipesegments = gg.generate_pipe_description(pipechain)
    for ps in pipesegments[1:-1]: 
        #find pos along length of pipe:
        ps['perc'] = (radius + (ps["length"] - 2.0 * radius) * random.random())/ps["length"]
        ps['exists'] = random.random() if (ps["length"] > (4.0*radius)) else 0
    pipesegments[0]['perc'] = 0.0
    pipesegments[0]['exists'] = 1.1
    pipesegments[-1]['perc'] = 1.0
    pipesegments[-1]['exists'] = 1.1
    
    new_obj = None
    for s in pipesegments:
        if (1-threshold) <= s['exists']:
            pos = s['start'] + (s['end'] - s['start']) * s['perc']
            new_obj = obj.copy()
            new_obj.rotation_mode = 'QUATERNION'
            new_obj.rotation_quaternion = s["orientation"]
            new_obj.location = pos
            #new_obj.data = src_obj.data.copy()
            #new_obj.animation_data_clear()
            #bpy.context.scene.collection.objects.link(new_obj)
            objs.append(new_obj)
    
    return objs
    
def render_supports(pipechain, support_obj, radius, period, obj_catalog):
    if period == 0:  return []
    newobjs = []
    
    #TODO: optimizae and take this function away to generate pipe description
    #only once
    pipesegments = gg.generate_pipe_description(pipechain)
    
    verts_clamp = gu.createcylinder(radius*1.2, 
                                    b1 = -0.2*radius, b2 = 0.2*radius,
                                    res = 10, closed=(1, 1))
    obj_clamp = gu.genobject("clamp",verts_clamp)
    verts_support = gu.createcylinder(radius*0.2, 
                                    b1 = 0, b2 = 1.0,
                                    res = 3, closed=(0, 0))
    obj_support = gu.genobject("support",verts_support)
    
    obj_clamp = obj_catalog.append_if_not_exist("clamp", obj_clamp)
    obj_support = obj_catalog.append_if_not_exist("support", obj_support)
    
    #generate a list of interfaces for which a support is needed
    #TODO: replace interfaces with an "interface" class
    interfaces = []
    length = 0.0
    for ps in pipesegments[2:-2]:
        #first find a location for the support
        # check if the 
        pos = ps['start'] + ps['vec'] * 0.5
        pipe_dir = ps['vec']
        if (length - period) > 0:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            _,location,_,_ = support_obj.closest_point_on_mesh(pos)
            support_dir = mh.vec(location) - pos
            support_len = mh.norm(support_dir)
            cosang = mh.calc_angle_vec(pipe_dir,support_dir) * mh.rad
            cosang = abs(cosang - 90.0)

            #print(cosang)
            if cosang < 45.0 * mh.deg:
                #get orientation of pipe
                quat = mh.getquatrot((0,0,1.0),support_dir)
                
                new_obj = obj_support.copy()
                new_obj.rotation_mode = 'QUATERNION'
                new_obj.rotation_quaternion = quat#ps["orientation"]
                new_obj.location = pos
                new_obj.scale = (1.0,1.0,support_len*1.1)
                #new_obj.data = src_obj.data.copy()
                #new_obj.animation_data_clear()
                #bpy.context.scene.collection.objects.link(new_obj)
                newobjs.append(new_obj)
                
                new_obj = obj_clamp.copy()
                new_obj.rotation_mode = 'QUATERNION'
                new_obj.rotation_quaternion = ps["orientation"]
                new_obj.location = pos
                #new_obj.data = src_obj.data.copy()
                #new_obj.animation_data_clear()
                #bpy.context.scene.collection.objects.link(new_obj)
                newobjs.append(new_obj)
        else:
            length -= period
        
        length += ps['length']
        
        #start = p1,
        #end = p2,
        #vec = s1,
        #orientation = q,
        #x_start = x_start,
        #x_end = x_end,
        #length = mh.norm(s1),
        #angle_end = angle_end*0.5,
        #angle_start = angle_start*0.5,
        #twist_end = twist_end)
            
    
    #for ps in pipesegments[1:-1]: 
    #    #find pos along length of pipe:
    #    ps['perc'] = (radius + (ps["length"] - 2.0 * radius) * random.random())/ps["length"]
    #    ps['exists'] = random.random() if (ps["length"] > (2.0*radius)) else 0
    
    
    return newobjs
    

def render_curve(pipechain, mode, radius, res_v): 
    pipe_orig = mh.vec3(0,0,0)#pipechain[1]
    #pipesegments = gg.generate_pipe_description(pipechain)      
    
    #convert to edges
    edges = []
    for i in range(len(pipechain)-1):
        edges += [[pipechain[i], pipechain[i+1]]]

    if mode == 'skin':
        obj = gu.genobject("pipe",edges)
        #o2.rotation_mode = 'QUATERNION'
        #o2.rotation_quaternion = q
    
        mod = obj.modifiers.new("skin_piperator", 'SKIN')
        mod.use_smooth_shade = True
    
        for v in obj.data.skin_vertices[0].data:
            v.radius = radius, radius

        mod = obj.modifiers.new('subsurf1_piperator','SUBSURF')
        #res_v = (4 * 2**levels) -> levels = math.log2(res_v/4)
        levels = math.log2(res_v/4)
        mod.levels = levels
        mod.render_levels = levels

    elif mode == 'polycurve':
        crv = bpy.data.curves.new('pipe', type='CURVE')
        crv.dimensions = '3D'
        crv.resolution_u = 10
        crv.bevel_depth = radius
        #res = 4+2*num -> num = (res - 4)/2
        crv.bevel_resolution = (res_v - 4) / 2
        crv.extrude = 0.0 #taper extrude (makes curve being extracted "vertically")
        #crv.twist_mode = 
        #crv.twist_smooth = 
        
        # make a new spline in that curve
        spline = crv.splines.new(type='POLY')
        # a spline point for each point
        spline.points.add(len(pipechain)-1) # theres already one point by default
        # assign the point coordinates to the spline points
        node_weight = 1.0
        for p, new_co in zip(spline.points, pipechain):
            coords = (new_co.tolist() + [node_weight])
            #print(coords,type(coords))
            p.co = coords
            #TODO: radius should be multiplied if there is an "angle"
            # in the pipe
            p.radius = 1.0 #with bevel this acts basically as a multiplier
            #TODO: p.tilt = tilt
        
        #spline.order_u --> resolution for bezier & nurbs curves
        #spline.resolution_u --> resolution in u direction (along the curve)
            
        # make a new object with the curve
        obj = bpy.data.objects.new('pipe', crv)
    else:
        obj = gu.genobject("pipe",edges)        

    obj.location = pipe_orig#calc_polygon_center    
    return obj

def add_pipe(radius, debug = False):
    interfaces = get_interfaces()

    pipechain, pipe_orig = gg.generate_pipe_wireframe(interfaces)

    render_pipes(pipechain)

def add_pipes(sel_object,
              radius, offset, number, seed = 1, debug = 0,
              surfaceglue = True, mode = 'pipes',
              flange_appearance = 0.0,
              res_v = 10.0, material_index = 0,
              support_period = 0.3):

    #TODO: put all of the stuff below into the execute function of the operator
    # to make the parameter passing simpler
    
    printinfo(f'mode: {mode}')
    
    #check if object has an associated collection 
    if "pipe_collection" in sel_object.keys():
        if sel_object['pipe_collection'] in bpy.data.collections.keys():
            newcol = bpy.data.collections[sel_object['pipe_collection']]
        else:
            newcol = bpy.data.collections.new('pipes')
            bpy.context.scene.collection.children.link(newcol)
            sel_object['pipe_collection'] = newcol.name
    else:
        newcol = bpy.data.collections.new('pipes')
        bpy.context.scene.collection.children.link(newcol)
        sel_object['pipe_collection'] = newcol.name
    
    i = 0
    counter = 0
    random.seed(seed)
    pathmesh = pathfinder()
    #pathmesh_geometry = generate_pathfinding_mesh_from_slices(sel_object, interfaces, offset)
    pathmesh.generate_pathfinding_mesh(sel_object, [offset, radius*2.0*1.1])
    interfaces = get_interfaces_from_object_faces(sel_object)
    intf_num = len(interfaces)
    random.shuffle(interfaces)
    occupied_interfaces = []
    max_search_tries = 20
    obj_catalog = object_catalog()
    if len(sel_object.material_slots) > material_index:
        pipe_material = sel_object.material_slots[material_index].material
    else: pipe_material = None
    while (i < number) and (counter < max_search_tries):
        try:
            if len(interfaces) < 2: 
                infomsg = f'probably not enough free faces available ({intf_num})!'
                printinfo(infomsg)
                return infomsg
            ints = interfaces[0:2]
            #import ipdb; ipdb.set_trace()
            wire_verts, state = create_wires_around_mesh(sel_object, 
                                                  ints,
                                                    offset = offset,
                                                    pathmesh = pathmesh,
                                                    surfaceglue = surfaceglue)
            
            if state == 'occupied':
                if wire_verts[0]: occupied_interfaces+=interfaces.pop(0)
                if wire_verts[1]: occupied_interfaces+=interfaces.pop(1)
                printinfo(f"occupied vertices found, removing them from list {wire_verts}")
                continue
            
            if mode=='pipes':
                newobj = render_pipes(wire_verts, radius = radius, debug = debug)
            else:
                newobj = render_curve(wire_verts, mode, radius, res_v)
                
            newobjs = render_components(wire_verts, radius, mode,
                                        [flange_appearance],obj_catalog)
            
            support_objs = render_supports(wire_verts, sel_object,
                                           radius,
                                           period = support_period,
                                           obj_catalog = obj_catalog)
            
            for no in newobjs + support_objs: 
                no['piperator_id']=i
                no.parent = sel_object
                newcol.objects.link(no)
            #this only has to be done once, because the objects meshes
            # are only linked and therefore share the materials
            if len(newobjs) > 0 : gu.assign_material(no.data,pipe_material)
            
            newobj['piperator_id']=i
            newobj.parent = sel_object
            gu.assign_material(newobj.data,pipe_material)
            #newobj.data.materials.append(pipe_material)
            newcol.objects.link(newobj)
            i+=1
            interfaces = interfaces[2:]
            counter = 0
        except nx.exception.NetworkXNoPath:
            #TODO: put this inthe ceate_wires_around_mesh function for
            #        better consistency and code layout
            #logger.exception
            infomsg = "did not find a path, trying again and reshuffle interfaces"
            printinfo(infomsg)
            #reshuffle, maybe this makes it easier to find new working combinations
            random.shuffle(interfaces)
            counter += 1
        except IndexError:
            infomsg = f'probably not enough faces available ({intf_num})!'
            printinfo(infomsg)
            return infomsg
#        except:
#            logger.exception("unknown exception!")
#            break
    if counter == max_search_tries:
        msg = f"no more paths found ({i} of {number})"
        printinfo(msg)
        return msg
    
    
    for obj in bpy.data.objects: obj.select_set(False)
    sel_object.select_set(True)
    
    return "success"

from bpy.props import (
    BoolProperty,
    BoolVectorProperty,
    EnumProperty,
    IntProperty,
    FloatProperty,
    FloatVectorProperty,
)


class AddPipe(bpy.types.Operator):
    """Add pipes on faces of a mesh"""
    bl_idname = "mesh.add_pipes"
    bl_label = "Add Pipes"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(
        name = "mesh mode",
        description = "choose mesh generation mode",
        items={('pipes', 'Pipe Mesh', 'Generate pipe meshes'),
            ('polycurve', 'Poly Curve Object', 'Use curve objects with poly splines and simple bevel for pipes'),
            ('wire', 'Simple Wire', 'Use simple wireframe'),
            ('skin', 'Skin Modifier', 'Use simple wireframe with skin modifier')},
        default='skin'
        )
        
    flange_appearance: FloatProperty(
        name="flange_probability",
        description="probability of a flange appearing on each segment",
        min = 0.0, max = 1.0,
        step = 1.0,
        default = 0.3
    )
    
    support_period: FloatProperty(
        name="support period",
        description="period length of supports",
        min = 0.0, max = 10000,
        step = 1.0,
        default = 0.3
    )
    
    radius: FloatProperty(
        name="radius",
        description="radius of pipes",
        min=0.001, max=1000.0,
        step = 1.0,
        default=.05,
    )

    material_idx: IntProperty(name="material slot index",
        description="Material slot index from Source object",
        default=0,
        min = 0,
    )
    
    res_v: IntProperty(name="resolution v",
        description="resolution of pipe circumference",
        default=10,
        min = 4,
    )
    
    offset: FloatProperty(name="offset",
        description="offset from mesh",
        min=-1000, max=1000.0,
        step = 1.0,
        default=.11,
    )
    
    seed: IntProperty(name="random seed",
        description="seed value for randomness",
        default=10,
    )
    
    number: IntProperty(name="number of pipes",
        description="number of pipes",
        min = 0,
        default = 2
    )

    surfaceglue: BoolProperty(name="glue to surface",
        description="make pipes stay as close as possible to surface",
        default=True,
    )
    
    reset: BoolProperty(name="reset",
        description="delete previously created pipes",
        default=True,
    )

    debug: IntProperty(name="debug mode",
        description="debug mode", default=0,
    )

    def execute(self, context):
        if self.reset:
            for p_ob in context.selected_objects:
              for c_ob in p_ob.children:
                if 'piperator_id' in c_ob.keys():
                    bpy.data.objects.remove(c_ob, do_unlink=True)            

        #TODO: enable "poll" method for better object checking

        if len(bpy.context.selected_objects) == 0:
            printinfo('No objects selected!')
            return {'CANCELLED'}
        sel_object = bpy.context.selected_objects[0]
        if sel_object.type != 'MESH':
            printinfo("Wrong object type!")
            return {'CANCELLED'}

        #add_pipe(radius = self.radius)
        state = add_pipes(sel_object,
                          radius = self.radius,
                    offset = self.offset,
                    number = self.number,
                    seed = self.seed,
                    debug = self.debug,
                    surfaceglue = self.surfaceglue,
                    mode = self.mode,
                    flange_appearance = self.flange_appearance,
                    res_v = self.res_v,
                    material_index = self.material_idx,
                    support_period = self.support_period)        

        if state != "success":
            #self.report({'INFO'}, state)
            render_components.catalog = object_catalog()
            self.report({'ERROR'}, state)
            return {'CANCELLED'}
            
        return {'FINISHED'}

        #TODO: implement function
        """def draw(self, context):
                layout = self.layout
                layout.prop(self, "val1")
                
                if self.val1:
                    box = layout.box()
                    box.prop(self, "val2")
                    box.prop(self, "val3")
        """

def menu_func(self, context):
    self.layout.operator(AddPipe.bl_idname, icon='META_CAPSULE')


def register():
    bpy.utils.register_class(AddPipe)
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
    bpy.utils.unregister_class(AddPipe)
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)


if __name__ == "__main__":
    #logging.basicConfig(level=logging.INFO)     #already set in blender this way
    #unregister()
    register()
    import pdb, traceback, sys
    import trimesh
    try:
        
        #get selected object
        """ob = bpy.context.selected_objects[0]
        
        vs, es = generate_pathfinding_mesh(ob)
        
        newobj = gu.genobjfrompydata(verts = vs,
                                 edges = es)"""
        # test call
        bpy.ops.mesh.add_pipes()
        pass
    except:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        #pdb.post_mortem(tb)
