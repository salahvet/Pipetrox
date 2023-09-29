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


__author__ = "yeus <Thomas.Meschede@web.de>"
__status__ = "test"
__version__ = "0.09"
__date__ = "2019 Oct 23rd"

bl_info = {
    "name": "Piperator",
    "author": "yeus <Thomas.Meschede@web.de",
    "version": (0, 0, 9),
    "blender": (2, 80, 0),
    "location": "View3D > Add > Mesh > Add Pipes",
    "description": "Generates Pipes between selected faces",
    "warning": "",
    "wiki_url": "",
    "category": "Add Mesh",
    "support": "TESTING",
}



import bpy
#from operator import * 

#TODO: remove tabs
#https://blender.stackexchange.com/questions/97502/removing-tabs-from-tool-shelf-t-key/97503#97503

import os, sys
#
#blend_dir = os.path.dirname(bpy.data.filepath)
#if blend_dir not in sys.path:
#   sys.path.append(blend_dir)
# temporarily appends the folder containing this file into sys.path
main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib')
sys.path.append(main_dir)

import networkx
from . import pipe_operator
import importlib
importlib.reload(pipe_operator)


class piperator_delete(bpy.types.Operator):
    """Delete all pipes in the scene"""
    bl_idname = "object.piperator_delete"
    bl_label = "Delete All Piperator Objects"

    #operator functions
    # https://blender.stackexchange.com/questions/19416/what-do-operator-methods-do-poll-invoke-execute-draw-modal

    #@classmethod
    #def poll(cls, context):
    #    return context.active_object is not None

    def execute(self, context):
        for ob in context.scene.objects:
            if 'piperator_id' in ob.keys():
                bpy.data.objects.remove(ob, do_unlink=True)
        return {'FINISHED'}

class piperator_delete_children(bpy.types.Operator):
    """Delete all pipes linked to this object"""
    bl_idname = "object.piperator_delete_children"
    bl_label = "Delete Object Pipes"

    def execute(self, context):
        for p_ob in context.selected_objects:
          for c_ob in p_ob.children:
            if 'piperator_id' in c_ob.keys():
                bpy.data.objects.remove(c_ob, do_unlink=True)
        return {'FINISHED'}

class piperator_panel(bpy.types.Panel):
    """Creates a Panel in the scene context of the properties editor"""
    bl_label = "Piperator"
    bl_idname = "SCENE_PT_layout"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'piperator'
    #bl_context = "tool"

    def draw(self, context):
        layout = self.layout

        scene = context.scene

        layout.operator("mesh.add_pipes")
        layout.operator("object.piperator_delete_children")
        layout.operator("object.piperator_delete")
        
        """
        # Create a simple row.
        layout.label(text=" Simple Row:")

        row = layout.row()
        row.prop(scene, "frame_start")
        row.prop(scene, "frame_end")

        # Create an row where the buttons are aligned to each other.
        layout.label(text=" Aligned Row:")

        row = layout.row(align=True)
        row.prop(scene, "frame_start")
        row.prop(scene, "frame_end")

        # Create two columns, by using a split layout.
        split = layout.split()

        # First column
        col = split.column()
        col.label(text="Column One:")
        col.prop(scene, "frame_end")
        col.prop(scene, "frame_start")

        # Second column, aligned
        col = split.column(align=True)
        col.label(text="Column Two:")
        col.prop(scene, "frame_start")
        col.prop(scene, "frame_end")

        # Big render button
        layout.label(text="Big Button:")
        row = layout.row()
        row.scale_y = 3.0
        row.operator("render.render")

        # Different sizes in a row
        layout.label(text="Different button sizes:")
        row = layout.row(align=True)
        row.operator("render.render")

        sub = row.row()
        sub.scale_x = 2.0
        sub.operator("render.render")

        row.operator("render.render")
        """

classes = (
    piperator_panel,
    piperator_delete,
    piperator_delete_children
)
register_panel, unregister_panel = bpy.utils.register_classes_factory(classes)

def register():
    pipe_operator.register()
    register_panel()

def unregister():
    pipe_operator.unregister()
    unregister_panel()

if __name__ == "__main__":
    pipe_operator.register()
    #bpy.ops.mesh.add_pipes()
    register()
    
    #debugging
    #bpy.ops.mesh.add_pipes(number=5, mode='skin', seed = 11)
    
#obj = bpy.context.selected_objects[0]
