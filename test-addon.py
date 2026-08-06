bl_info = {
    "name": "Test Addon",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar",
    "description": "A test add-on for Blender.",
    "warning": "",
    "wiki_url": "",
    "tracker_url": "",
    "category": "Object"
}

import bpy

class OBJECT_OT_test_addon(bpy.types.Operator):
    """Tooltip"""
    bl_idname = "object.test_addon"
    bl_label = "Test Addon"

    def execute(self, context):
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
        return {'FINISHED'}

class TestAddonPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "TestAddon"
    bl_idname = "OBJECT_PT_test_addon"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "TestAddonCategory"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.test_addon")

def register():
    try:
        print("Registering Test Addon")
        bpy.utils.register_class(OBJECT_OT_test_addon)
        bpy.utils.register_class(TestAddonPanel)
    except Exception as e:
        print(f"Failed to register Test Addon: {e}")

def unregister():
    try:
        print("Unregistering Test Addon")
        bpy.utils.unregister_class(OBJECT_OT_test_addon)
        bpy.utils.unregister_class(TestAddonPanel)
    except Exception as e:
        print(f"Failed to unregister Test Addon: {e}")

if __name__ == "__main__":
    register()