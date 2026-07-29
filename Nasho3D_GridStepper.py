bl_info = {
    "name": "Grid Scale Hotkeys",
    "author": "ChatGPT",
    "version": (1, 4),
    "blender": (3, 0, 0),
    "location": "3D View",
    "description": "Increase/decrease grid scale using [ and ]",
    "category": "3D View",
}

import bpy
import blf

addon_keymaps = []
draw_handle = None

grid_message = ""
message_visible = False


# ---------------------------------------------------
# Draw Text
# ---------------------------------------------------

def draw_callback_px():

    if not message_visible:
        return

    font_id = 0

    region = bpy.context.region

    if region is None:
        return

    blf.size(font_id, 24)

    text_width = blf.dimensions(font_id, grid_message)[0]

    x = (region.width - text_width) / 2
    y = region.height - 50

    blf.position(font_id, x, y, 0)
    blf.draw(font_id, grid_message)


# ---------------------------------------------------
# Redraw
# ---------------------------------------------------

def redraw_all_viewports():

    wm = bpy.context.window_manager

    for window in wm.windows:
        for area in window.screen.areas:

            if area.type == 'VIEW_3D':
                area.tag_redraw()


# ---------------------------------------------------
# Hide Message
# ---------------------------------------------------

def hide_message():

    global message_visible

    message_visible = False

    redraw_all_viewports()

    return None


# ---------------------------------------------------
# Change Grid Scale
# ---------------------------------------------------

def change_grid_scale(multiplier):

    global grid_message
    global message_visible

    wm = bpy.context.window_manager

    for window in wm.windows:

        screen = window.screen

        for area in screen.areas:

            if area.type == 'VIEW_3D':

                for space in area.spaces:

                    if space.type == 'VIEW_3D':

                        current = space.overlay.grid_scale

                        new_value = current * multiplier

                        new_value = max(0.001, new_value)

                        space.overlay.grid_scale = new_value

                        grid_message = f"Grid Scale: {new_value:g}"

                        message_visible = True

    redraw_all_viewports()

    bpy.app.timers.register(
        hide_message,
        first_interval=2.0
    )


# ---------------------------------------------------
# Operators
# ---------------------------------------------------

class VIEW3D_OT_grid_scale_increase(bpy.types.Operator):
    bl_idname = "view3d.grid_scale_increase"
    bl_label = "Increase Grid Scale"

    def execute(self, context):

        change_grid_scale(2.0)

        return {'FINISHED'}


class VIEW3D_OT_grid_scale_decrease(bpy.types.Operator):
    bl_idname = "view3d.grid_scale_decrease"
    bl_label = "Decrease Grid Scale"

    def execute(self, context):

        change_grid_scale(0.5)

        return {'FINISHED'}


# ---------------------------------------------------
# Register
# ---------------------------------------------------

classes = (
    VIEW3D_OT_grid_scale_increase,
    VIEW3D_OT_grid_scale_decrease,
)


def register():

    global draw_handle

    for cls in classes:
        bpy.utils.register_class(cls)

    # Draw handler
    draw_handle = bpy.types.SpaceView3D.draw_handler_add(
        draw_callback_px,
        (),
        'WINDOW',
        'POST_PIXEL'
    )

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon

    if kc:

        keymap_names = [
            "Object Mode",
            "Mesh",
        ]

        for km_name in keymap_names:

            km = kc.keymaps.new(
                name=km_name,
                space_type='VIEW_3D'
            )

            # ]
            kmi = km.keymap_items.new(
                "view3d.grid_scale_increase",
                'RIGHT_BRACKET',
                'PRESS'
            )

            addon_keymaps.append((km, kmi))

            # [
            kmi = km.keymap_items.new(
                "view3d.grid_scale_decrease",
                'LEFT_BRACKET',
                'PRESS'
            )

            addon_keymaps.append((km, kmi))


def unregister():

    global draw_handle

    if draw_handle is not None:

        bpy.types.SpaceView3D.draw_handler_remove(
            draw_handle,
            'WINDOW'
        )

        draw_handle = None

    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)

    addon_keymaps.clear()

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()