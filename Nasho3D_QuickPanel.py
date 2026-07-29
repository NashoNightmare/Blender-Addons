bl_info = {
    "name": "Oshan Tools Quick Panel",
    "author": "Oshan",
    "version": (1, 3, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Oshan Tools",
    "description": "Adds a custom side panel in the 3D Viewport UI, updateable via addon preferences, and includes a lattice deformer generator.",
    "category": "3D View",
}

import bpy
import os
import shutil
import pathlib
import mathutils

# =========================================================================
# HELPER FUNCTIONS
# =========================================================================
def get_preferences(context):
    """Safely retrieves the addon preferences for this module."""
    addon_name = __package__ or __name__
    if addon_name in context.preferences.addons:
        return context.preferences.addons[addon_name].preferences
    
    # Fallback to filename stem if registered under a different name
    filename = pathlib.Path(__file__).stem
    if filename in context.preferences.addons:
        return context.preferences.addons[filename].preferences
        
    return None


def create_lattice_for_object(obj, interpolation_u, interpolation_v, interpolation_w, res_u, res_v, res_w):
    """Creates a lattice matching the object's local bounding box exactly and adds a Lattice modifier."""
    # 1. Get local bounding box center and size
    local_bbox = [mathutils.Vector(corner) for corner in obj.bound_box]
    min_coords = mathutils.Vector((min(c[0] for c in local_bbox), min(c[1] for c in local_bbox), min(c[2] for c in local_bbox)))
    max_coords = mathutils.Vector((max(c[0] for c in local_bbox), max(c[1] for c in local_bbox), max(c[2] for c in local_bbox)))
    
    center_local = (min_coords + max_coords) / 2
    size_local = max_coords - min_coords
    
    # 2. Create lattice data block and object
    name = f"{obj.name}_Lattice"
    lat_data = bpy.data.lattices.new(name)
    lat_obj = bpy.data.objects.new(name, lat_data)
    
    # 3. Set interpolation types
    lat_data.interpolation_type_u = interpolation_u
    lat_data.interpolation_type_v = interpolation_v
    lat_data.interpolation_type_w = interpolation_w
    
    # 4. Set resolution (must be minimum of 2)
    lat_data.points_u = max(2, res_u)
    lat_data.points_v = max(2, res_v)
    lat_data.points_w = max(2, res_w)
    
    # 5. Link to active collection
    bpy.context.collection.objects.link(lat_obj)
    
    # The default lattice data vertices span from -0.5 to 0.5 (local size of 1.0).
    # So we translate by center_local and scale by size_local.
    scale_local = size_local.copy()
    
    # Avoid zero thickness to prevent degenerate scaling matrices
    for i in range(3):
        if scale_local[i] < 0.0001:
            scale_local[i] = 0.0001
            
    # Construct local matrix for the lattice relative to the object
    mat_local = mathutils.Matrix.Translation(center_local) @ mathutils.Matrix.Diagonal(scale_local.to_4d())
    
    # Apply object's world matrix to place the lattice in the correct world space location/rotation/scale
    lat_obj.matrix_world = obj.matrix_world @ mat_local
    
    # 6. Add Lattice modifier to the target object
    mod = obj.modifiers.new(name="Lattice", type='LATTICE')
    mod.object = lat_obj
    
    return lat_obj


def get_uv_islands(bm, uv_layer):
    """Finds all UV islands in the BMesh and returns them as a list of lists of loops."""
    # Gather all loops in BMesh
    loops = []
    for face in bm.faces:
        loops.extend(face.loops)

    # DSU (Disjoint Set Union) helper structures
    parent = {l: l for l in loops}

    def find(l):
        path = []
        while parent[l] != l:
            path.append(l)
            l = parent[l]
        for node in path:
            parent[node] = l
        return l

    def union(l1, l2):
        r1 = find(l1)
        r2 = find(l2)
        if r1 != r2:
            parent[r1] = r2

    # 1. Connect all loops within the same face
    for face in bm.faces:
        if len(face.loops) > 1:
            first = face.loops[0]
            for l in face.loops[1:]:
                union(first, l)

    # 2. Connect loops sharing the same mesh vertex and UV coordinates (welded)
    for vert in bm.verts:
        if len(vert.link_loops) > 1:
            uv_groups = []
            for l in vert.link_loops:
                uv = l[uv_layer].uv
                found = False
                for grp in uv_groups:
                    ref_l = grp[0]
                    ref_uv = ref_l[uv_layer].uv
                    if (uv - ref_uv).length < 1e-5:
                        grp.append(l)
                        found = True
                        break
                if not found:
                    uv_groups.append([l])
            
            # Union loops within each UV-welded group
            for grp in uv_groups:
                first = grp[0]
                for other in grp[1:]:
                    union(first, other)

    # Group loops by their DSU representative parent
    from collections import defaultdict
    islands = defaultdict(list)
    for l in loops:
        islands[find(l)].append(l)

    return list(islands.values())


# =========================================================================
# FUTURE OPERATORS GO HERE
# =========================================================================
# When you want to add new button actions (operators):
# 1. Define an Operator class like the commented example below.
# 2. Add the class name to the `classes` tuple at the bottom of the script.
# 3. Add a layout.operator() call in the Panel's draw method.
#
# Example Operator:
# class OBJECT_OT_my_custom_operator(bpy.types.Operator):
#     """Tooltip info for my custom operator"""
#     bl_idname = "object.my_custom_operator"
#     bl_label = "My Custom Operator"
#     bl_options = {'REGISTER', 'UNDO'}
# 
#     def execute(self, context):
#         # Your custom Python code goes here
#         self.report({'INFO'}, "Operator Executed Successfully!")
#         return {'FINISHED'}


# =========================================================================
# UV TOOLS OPERATORS
# =========================================================================
class NASH3D_OT_snap_to_coordinate(bpy.types.Operator):
    """Rotate the selected UV island by 90-degree increments to place the selected vertex at the bottom-left, then move it to the target coordinates"""
    bl_idname = "nash3d.snap_to_coordinate"
    bl_label = "Snap to coordinate"
    bl_options = {'REGISTER', 'UNDO'}

    # Target snap coordinates
    target_u: bpy.props.FloatProperty(
        name="Target U",
        description="U coordinate to snap the selected vertex to",
        default=0.0
    )
    target_v: bpy.props.FloatProperty(
        name="Target V",
        description="V coordinate to snap the selected vertex to",
        default=1.0
    )

    @classmethod
    def poll(cls, context):
        return (context.active_object and 
                context.active_object.type == 'MESH' and 
                context.active_object.mode == 'EDIT')

    def execute(self, context):
        import bmesh
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            self.report({'ERROR'}, "No active UV layer found.")
            return {'CANCELLED'}
            
        islands = get_uv_islands(bm, uv_layer)
        
        processed_count = 0
        for island in islands:
            sel_loops = [l for l in island if (l.uv_select_vert if hasattr(l, "uv_select_vert") else l[uv_layer].select)]
            if not sel_loops:
                continue
                
            # Use the first selected loop as the reference point for rotation
            ref_loop = sel_loops[0]
            u_sel = ref_loop[uv_layer].uv.x
            v_sel = ref_loop[uv_layer].uv.y
            
            # Find the best rotation angle out of 0, 90, 180, 270 degrees
            best_angle = 0
            min_dist_sq = float('inf')
            
            # 0 degrees
            min_u_0 = min(l[uv_layer].uv.x for l in island)
            min_v_0 = min(l[uv_layer].uv.y for l in island)
            d0 = (u_sel - min_u_0)**2 + (v_sel - min_v_0)**2
            if d0 < min_dist_sq:
                min_dist_sq = d0
                best_angle = 0
                
            # 90 degrees CCW (rotated around u_sel, v_sel)
            min_u_90 = min(u_sel - (l[uv_layer].uv.y - v_sel) for l in island)
            min_v_90 = min(v_sel + (l[uv_layer].uv.x - u_sel) for l in island)
            d90 = (u_sel - min_u_90)**2 + (v_sel - min_v_90)**2
            if d90 < min_dist_sq:
                min_dist_sq = d90
                best_angle = 90
                
            # 180 degrees
            min_u_180 = min(u_sel - (l[uv_layer].uv.x - u_sel) for l in island)
            min_v_180 = min(v_sel - (l[uv_layer].uv.y - v_sel) for l in island)
            d180 = (u_sel - min_u_180)**2 + (v_sel - min_v_180)**2
            if d180 < min_dist_sq:
                min_dist_sq = d180
                best_angle = 180
                
            # 270 degrees CCW
            min_u_270 = min(u_sel + (l[uv_layer].uv.y - v_sel) for l in island)
            min_v_270 = min(v_sel - (l[uv_layer].uv.x - u_sel) for l in island)
            d270 = (u_sel - min_u_270)**2 + (v_sel - min_v_270)**2
            if d270 < min_dist_sq:
                min_dist_sq = d270
                best_angle = 270
                
            # Apply the best rotation and translation directly to loops using target coordinates
            for l in island:
                u, v = l[uv_layer].uv.x, l[uv_layer].uv.y
                if best_angle == 0:
                    l[uv_layer].uv.x = u - u_sel + self.target_u
                    l[uv_layer].uv.y = v - v_sel + self.target_v
                elif best_angle == 90:
                    l[uv_layer].uv.x = v_sel - v + self.target_u
                    l[uv_layer].uv.y = u - u_sel + self.target_v
                elif best_angle == 180:
                    l[uv_layer].uv.x = u_sel - u + self.target_u
                    l[uv_layer].uv.y = v_sel - v + self.target_v
                elif best_angle == 270:
                    l[uv_layer].uv.x = v - v_sel + self.target_u
                    l[uv_layer].uv.y = u_sel - u + self.target_v
                    
            processed_count += 1
            
        if processed_count > 0:
            bmesh.update_edit_mesh(obj.data)
            self.report({'INFO'}, f"Aligned and snapped {processed_count} island(s) to ({self.target_u:.2f}, {self.target_v:.2f}).")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "No selected UV vertex/loops found in any island.")
            return {'CANCELLED'}


class NASH3D_OT_snap_to_vertex(bpy.types.Operator):
    """Snap one selected island to another based on two selected vertices.
The island with the active face (last selected) will act as the target,
and the other island will snap to it."""
    bl_idname = "nash3d.snap_to_vertex"
    bl_label = "Snap to Vertex"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.active_object and 
                context.active_object.type == 'MESH' and 
                context.active_object.mode == 'EDIT')

    def execute(self, context):
        import bmesh
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            self.report({'ERROR'}, "No active UV layer found.")
            return {'CANCELLED'}
            
        selected_loops = []
        for face in bm.faces:
            for l in face.loops:
                if (l.uv_select_vert if hasattr(l, "uv_select_vert") else l[uv_layer].select):
                    selected_loops.append(l)
                    
        uv_groups = []
        for l in selected_loops:
            uv = l[uv_layer].uv
            found = False
            for grp in uv_groups:
                if (uv - grp[0][uv_layer].uv).length < 1e-5:
                    grp.append(l)
                    found = True
                    break
            if not found:
                uv_groups.append([l])
                
        if len(uv_groups) != 2:
            self.report({'ERROR'}, f"Please select exactly 2 UV vertices from different islands. (Found {len(uv_groups)})")
            return {'CANCELLED'}
            
        active_face = bm.faces.active
        
        target_group = uv_groups[0]
        source_group = uv_groups[1]
        
        if active_face:
            for grp in uv_groups:
                if any(l.face == active_face for l in grp):
                    target_group = grp
                    source_group = uv_groups[0] if uv_groups[1] == grp else uv_groups[1]
                    break
                    
        islands = get_uv_islands(bm, uv_layer)
        source_island = None
        target_island = None
        
        for island in islands:
            if source_group[0] in island:
                source_island = island
            if target_group[0] in island:
                target_island = island
                
        if not source_island or not target_island:
            self.report({'ERROR'}, "Could not determine islands for the selected vertices.")
            return {'CANCELLED'}
            
        if source_island == target_island:
            self.report({'ERROR'}, "Selected vertices belong to the same island. Please select vertices from two different islands.")
            return {'CANCELLED'}
            
        source_uv = source_group[0][uv_layer].uv
        target_uv = target_group[0][uv_layer].uv
        
        translation = target_uv - source_uv
        
        for l in source_island:
            l[uv_layer].uv += translation
            
        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'}, "Snapped Island A to Island B.")
        return {'FINISHED'}


class NASH3D_OT_rotate_uv_island(bpy.types.Operator):
    """Rotate the selected UV island by a specific angle"""
    bl_idname = "nash3d.rotate_uv_island"
    bl_label = "Rotate UV Island"
    bl_options = {'REGISTER', 'UNDO'}

    angle: bpy.props.FloatProperty(
        name="Angle",
        description="Rotation angle in degrees",
        default=15.0
    )
    
    pivot: bpy.props.EnumProperty(
        name="Pivot",
        description="Pivot point for rotation",
        items=[
            ('ISLAND_CENTER', "Island Center", ""),
            ('BBOX_CENTER', "BBox Center", ""),
            ('VERTEX', "Vertex", ""),
        ],
        default='VERTEX'
    )

    @classmethod
    def poll(cls, context):
        return (context.active_object and 
                context.active_object.type == 'MESH' and 
                context.active_object.mode == 'EDIT')

    def execute(self, context):
        import bmesh
        import math
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            self.report({'ERROR'}, "No active UV layer found.")
            return {'CANCELLED'}
            
        islands = get_uv_islands(bm, uv_layer)
        
        angle_rad = math.radians(self.angle)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        
        # Pre-calculate global BBOX center if needed
        global_u_sel = 0.0
        global_v_sel = 0.0
        if self.pivot == 'BBOX_CENTER':
            min_u, max_u = float('inf'), float('-inf')
            min_v, max_v = float('inf'), float('-inf')
            has_sel = False
            for island in islands:
                for l in island:
                    if (l.uv_select_vert if hasattr(l, "uv_select_vert") else l[uv_layer].select):
                        has_sel = True
                        uv = l[uv_layer].uv
                        min_u = min(min_u, uv.x)
                        max_u = max(max_u, uv.x)
                        min_v = min(min_v, uv.y)
                        max_v = max(max_v, uv.y)
            if has_sel:
                global_u_sel = (min_u + max_u) / 2.0
                global_v_sel = (min_v + max_v) / 2.0
            else:
                self.report({'WARNING'}, "No selected UV vertex/loops found.")
                return {'CANCELLED'}
        
        processed_count = 0
        for island in islands:
            sel_loops = [l for l in island if (l.uv_select_vert if hasattr(l, "uv_select_vert") else l[uv_layer].select)]
            if not sel_loops:
                continue
                
            u_sel, v_sel = 0.0, 0.0
            
            if self.pivot == 'ISLAND_CENTER':
                min_u, max_u = float('inf'), float('-inf')
                min_v, max_v = float('inf'), float('-inf')
                for l in island:
                    uv = l[uv_layer].uv
                    min_u = min(min_u, uv.x)
                    max_u = max(max_u, uv.x)
                    min_v = min(min_v, uv.y)
                    max_v = max(max_v, uv.y)
                u_sel = (min_u + max_u) / 2.0
                v_sel = (min_v + max_v) / 2.0
            elif self.pivot == 'BBOX_CENTER':
                u_sel = global_u_sel
                v_sel = global_v_sel
            elif self.pivot == 'VERTEX':
                u_sum = sum(l[uv_layer].uv.x for l in sel_loops)
                v_sum = sum(l[uv_layer].uv.y for l in sel_loops)
                u_sel = u_sum / len(sel_loops)
                v_sel = v_sum / len(sel_loops)
            
            for l in island:
                du = l[uv_layer].uv.x - u_sel
                dv = l[uv_layer].uv.y - v_sel
                l[uv_layer].uv.x = u_sel + du * cos_theta - dv * sin_theta
                l[uv_layer].uv.y = v_sel + du * sin_theta + dv * cos_theta
                
            processed_count += 1
            
        if processed_count > 0:
            bmesh.update_edit_mesh(obj.data)
            self.report({'INFO'}, f"Rotated {processed_count} island(s) by {self.angle}°.")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "No selected UV vertex/loops found in any island.")
            return {'CANCELLED'}


# =========================================================================
# TEXEL DENSITY OPERATOR
# =========================================================================
class NASH3D_OT_set_texel_density(bpy.types.Operator):
    """Scale selected UV islands so their texel density matches the target.
    Target TD = Texture Size (px) / Physical Size (m). Each island is scaled
    uniformly around its UV centroid."""
    bl_idname = "nash3d.set_texel_density"
    bl_label = "Set Texel Density"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.active_object and
                context.active_object.type == 'MESH' and
                context.active_object.mode == 'EDIT')

    pivot: bpy.props.EnumProperty(
        name="Pivot",
        description="Pivot point for scaling",
        items=[
            ('BBOX_CENTER', "BBox Center", "Scale around the bounding box center of the island"),
            ('VERTEX', "Vertex", "Scale around the selected vertex (midpoint if multiple)"),
        ],
        default='BBOX_CENTER'
    )

    # ---- private helpers ------------------------------------------------

    @staticmethod
    def _world_face_area(face, matrix_world):
        """Return the world-space area of a BMFace (handles n-gons via fan triangulation)."""
        world_verts = [matrix_world @ v.co for v in face.verts]
        area = 0.0
        v0 = world_verts[0]
        for i in range(1, len(world_verts) - 1):
            cross = (world_verts[i] - v0).cross(world_verts[i + 1] - v0)
            area += cross.length * 0.5
        return area

    @staticmethod
    def _uv_face_area(face_loops, uv_layer):
        """Return the UV-space area of a face using the shoelace formula."""
        uvs = [l[uv_layer].uv for l in face_loops]
        n = len(uvs)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += uvs[i].x * uvs[j].y - uvs[j].x * uvs[i].y
        return abs(area) * 0.5

    # ---- execute --------------------------------------------------------

    def execute(self, context):
        import bmesh
        import math

        scene = context.scene
        texture_size = int(scene.oshan_td_texture_size)   # pixels
        physical_size = scene.oshan_td_physical_size       # metres

        if physical_size <= 0.0:
            self.report({'ERROR'}, "Physical size must be greater than 0.")
            return {'CANCELLED'}

        # Target texel density in texels/metre
        target_td = texture_size / physical_size

        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active

        if not uv_layer:
            self.report({'ERROR'}, "No active UV layer found.")
            return {'CANCELLED'}

        matrix_world = obj.matrix_world
        islands = get_uv_islands(bm, uv_layer)

        processed = 0
        for island in islands:
            # Only process islands that have at least one selected loop/vert
            sel_loops = [
                l for l in island
                if (l.uv_select_vert if hasattr(l, "uv_select_vert") else l[uv_layer].select)
            ]
            if not sel_loops:
                continue

            # --- Group loops by face -------------------------------------
            face_loop_map = {}
            for l in island:
                fid = l.face.index
                if fid not in face_loop_map:
                    face_loop_map[fid] = []
                face_loop_map[fid].append(l)

            # --- Accumulate world and UV areas --------------------------
            total_world_area = 0.0
            total_uv_area = 0.0
            for loops in face_loop_map.values():
                total_world_area += self._world_face_area(loops[0].face, matrix_world)
                total_uv_area    += self._uv_face_area(loops, uv_layer)

            if total_world_area < 1e-12 or total_uv_area < 1e-12:
                continue

            # --- Compute scale factor ------------------------------------
            current_td = texture_size * math.sqrt(total_uv_area / total_world_area)
            scale = target_td / current_td

            # --- Find scaling pivot for island ------------------------------
            cx, cy = 0.0, 0.0
            
            if self.pivot == 'BBOX_CENTER':
                min_u = min(l[uv_layer].uv.x for l in island)
                max_u = max(l[uv_layer].uv.x for l in island)
                min_v = min(l[uv_layer].uv.y for l in island)
                max_v = max(l[uv_layer].uv.y for l in island)
                cx = (min_u + max_u) / 2.0
                cy = (min_v + max_v) / 2.0
            elif self.pivot == 'VERTEX':
                u_sum = sum(l[uv_layer].uv.x for l in sel_loops)
                v_sum = sum(l[uv_layer].uv.y for l in sel_loops)
                cx = u_sum / len(sel_loops)
                cy = v_sum / len(sel_loops)

            # --- Scale uniformly around pivot ------------------------
            for l in island:
                uv = l[uv_layer].uv
                uv.x = cx + (uv.x - cx) * scale
                uv.y = cy + (uv.y - cy) * scale

            processed += 1

        bmesh.update_edit_mesh(obj.data)

        if processed == 0:
            self.report({'WARNING'}, "No selected UV islands found.")
            return {'CANCELLED'}

        self.report(
            {'INFO'},
            f"Set texel density to {target_td:.2f} px/m "
            f"({texture_size}px / {physical_size}m) on {processed} island(s)."
        )
        return {'FINISHED'}


# =========================================================================
# LATTICE DEFORMER OPERATOR
# =========================================================================
class NASH3D_OT_create_lattice(bpy.types.Operator):
    """Create a lattice deformer modifier on the selected object"""
    bl_idname = "nash3d.create_lattice"
    bl_label = "Create Lattice Deformer"
    bl_options = {'REGISTER', 'UNDO'}

    # Operator properties for U, V, W interpolation
    interpolation_u: bpy.props.EnumProperty(
        name="U Interpolation",
        items=[
            ('KEY_BSPLINE', "B-Spline", "B-Spline interpolation"),
            ('KEY_LINEAR', "Linear", "Linear interpolation"),
            ('KEY_CARDINAL', "Cardinal", "Cardinal interpolation"),
            ('KEY_CATMULL_ROM', "Catmull-Rom", "Catmull-Rom interpolation"),
        ],
        default='KEY_LINEAR'
    )

    interpolation_v: bpy.props.EnumProperty(
        name="V Interpolation",
        items=[
            ('KEY_BSPLINE', "B-Spline", "B-Spline interpolation"),
            ('KEY_LINEAR', "Linear", "Linear interpolation"),
            ('KEY_CARDINAL', "Cardinal", "Cardinal interpolation"),
            ('KEY_CATMULL_ROM', "Catmull-Rom", "Catmull-Rom interpolation"),
        ],
        default='KEY_LINEAR'
    )

    interpolation_w: bpy.props.EnumProperty(
        name="W Interpolation",
        items=[
            ('KEY_BSPLINE', "B-Spline", "B-Spline interpolation"),
            ('KEY_LINEAR', "Linear", "Linear interpolation"),
            ('KEY_CARDINAL', "Cardinal", "Cardinal interpolation"),
            ('KEY_CATMULL_ROM', "Catmull-Rom", "Catmull-Rom interpolation"),
        ],
        default='KEY_LINEAR'
    )

    # Operator properties for U, V, W resolution
    resolution_u: bpy.props.IntProperty(
        name="U Resolution",
        description="Number of control points along the U axis",
        default=2,
        min=2,
        max=64
    )

    resolution_v: bpy.props.IntProperty(
        name="V Resolution",
        description="Number of control points along the V axis",
        default=2,
        min=2,
        max=64
    )

    resolution_w: bpy.props.IntProperty(
        name="W Resolution",
        description="Number of control points along the W axis",
        default=2,
        min=2,
        max=64
    )

    def execute(self, context):
        obj = context.active_object
        if not obj:
            self.report({'ERROR'}, "No active object selected.")
            return {'CANCELLED'}

        if obj.type not in {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}:
            self.report({'ERROR'}, "Active object must be a geometry type (Mesh, Curve, etc.).")
            return {'CANCELLED'}

        try:
            lat_obj = create_lattice_for_object(
                obj, 
                self.interpolation_u, 
                self.interpolation_v, 
                self.interpolation_w,
                self.resolution_u,
                self.resolution_v,
                self.resolution_w
            )
            self.report({'INFO'}, f"Lattice '{lat_obj.name}' created and modifier added to '{obj.name}'")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Failed to create lattice: {str(e)}")
            return {'CANCELLED'}


# =========================================================================
# GROUP TOGETHER OPERATOR
# =========================================================================
class NASH3D_OT_group_together(bpy.types.Operator):
    """Parent selected objects to a new empty object at the active object's origin"""
    bl_idname = "nash3d.group_together"
    bl_label = "Group Together"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        selected_objects = [obj for obj in context.selected_objects]
        if not selected_objects:
            self.report({'ERROR'}, "No objects selected.")
            return {'CANCELLED'}

        active_obj = context.active_object
        if not active_obj or active_obj not in selected_objects:
            active_obj = selected_objects[0]

        # Use world space translation of the active object's origin
        group_location = active_obj.matrix_world.to_translation()

        # Create empty object
        empty_obj = bpy.data.objects.new("Group", None)
        empty_obj.empty_display_type = 'PLAIN_AXES'
        
        # Link empty to the active collection
        active_col = context.collection
        active_col.objects.link(empty_obj)
        
        # Position the empty at the origin of the active object
        empty_obj.location = group_location

        # Force a dependency graph update so the new empty's location is evaluated before parenting
        context.view_layer.update()

        # Parent selected objects to the empty, maintaining world matrices so they do not shift
        for obj in selected_objects:
            if obj == empty_obj:
                continue
            
            # Store current world matrix
            world_matrix = obj.matrix_world.copy()
            obj.parent = empty_obj
            # Restore world matrix to keep visual position
            obj.matrix_world = world_matrix

            # Move object to the active collection if it's not already there
            if obj not in active_col.objects.values():
                active_col.objects.link(obj)
            # Unlink from all other collections
            for c in list(obj.users_collection):
                if c != active_col:
                    c.objects.unlink(obj)

        # Deselect everything and select only the new group empty
        for obj in context.selected_objects:
            obj.select_set(False)

        empty_obj.select_set(True)
        context.view_layer.objects.active = empty_obj

        self.report({'INFO'}, f"Grouped {len(selected_objects)} objects under '{empty_obj.name}' in the active collection")
        return {'FINISHED'}


# =========================================================================
# EASY BEVEL OPERATOR
# =========================================================================
class NASH3D_OT_easy_bevel(bpy.types.Operator):
    """Set bevel weight of selected edges to 1.0 and add a Bevel modifier"""
    bl_idname = "nash3d.easy_bevel"
    bl_label = "Bevel it!"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a Mesh.")
            return {'CANCELLED'}

        # Store original mode to return to it later
        original_mode = obj.mode
        
        try:
            # We must set mode to OBJECT to access and modify mesh attributes in Blender 4.0+
            if original_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            
            mesh = obj.data
            attr_name = "bevel_weight_edge"
            
            # Access or create the bevel weight attribute
            if attr_name not in mesh.attributes:
                bevel_attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='EDGE')
            else:
                bevel_attr = mesh.attributes[attr_name]
            
            # Set the weight of selected edges to 1.0
            count = 0
            for i, edge in enumerate(mesh.edges):
                if edge.select:
                    bevel_attr.data[i].value = 1.0
                    count += 1
            
            # Update mesh representation to ensure the viewport updates
            mesh.update()
            
            # Switch back to original mode
            if original_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode=original_mode)
                
            # Add a bevel modifier if none exists of type 'BEVEL'
            has_bevel = any(m.type == 'BEVEL' for m in obj.modifiers)
            if not has_bevel:
                mod = obj.modifiers.new(name="Bevel", type='BEVEL')
                mod.limit_method = 'WEIGHT'
                mod.segments = 2
                mod.harden_normals = True
                self.report({'INFO'}, f"Set bevel weight to 1.0 on {count} edge(s) and added Bevel modifier.")
            else:
                self.report({'INFO'}, f"Set bevel weight to 1.0 on {count} edge(s) (Bevel modifier already exists).")
                
            return {'FINISHED'}
            
        except Exception as e:
            # Safely restore mode in case of failure
            if obj.mode != original_mode:
                try:
                    bpy.ops.object.mode_set(mode=original_mode)
                except:
                    pass
            self.report({'ERROR'}, f"Failed to apply Easy Bevel: {str(e)}")
            return {'CANCELLED'}


# =========================================================================
# UNBEVEL OPERATOR
# =========================================================================
class NASH3D_OT_unbevel(bpy.types.Operator):
    """Set bevel weight of selected edges to 0.0"""
    bl_idname = "nash3d.unbevel"
    bl_label = "Unbevel It!"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a Mesh.")
            return {'CANCELLED'}

        # Store original mode to return to it later
        original_mode = obj.mode
        
        try:
            # We must set mode to OBJECT to access and modify mesh attributes in Blender 4.0+
            if original_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
            
            mesh = obj.data
            attr_name = "bevel_weight_edge"
            
            # Access the bevel weight attribute
            if attr_name in mesh.attributes:
                bevel_attr = mesh.attributes[attr_name]
                
                # Set the weight of selected edges to 0.0
                count = 0
                for i, edge in enumerate(mesh.edges):
                    if edge.select:
                        bevel_attr.data[i].value = 0.0
                        count += 1
                
                # Update mesh representation to ensure the viewport updates
                mesh.update()
                
                info_msg = f"Set bevel weight to 0.0 on {count} edge(s)."
            else:
                info_msg = "No edge bevel weights found to clear."
            
            # Switch back to original mode
            if original_mode != 'OBJECT':
                bpy.ops.object.mode_set(mode=original_mode)
                
            self.report({'INFO'}, info_msg)
            return {'FINISHED'}
            
        except Exception as e:
            # Safely restore mode in case of failure
            if obj.mode != original_mode:
                try:
                    bpy.ops.object.mode_set(mode=original_mode)
                except:
                    pass
            self.report({'ERROR'}, f"Failed to apply Unbevel: {str(e)}")
            return {'CANCELLED'}


# =========================================================================
# COLOR SWATCH
# =========================================================================
class NASH3D_RecentColorItem(bpy.types.PropertyGroup):
    """Holds a single RGBA color entry for the color swatch."""
    color: bpy.props.FloatVectorProperty(
        name="Color",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0)
    )


def push_recent_color(scene, color_tuple):
    """Push a color into the scene's color swatch ring buffer (max 5).
    Duplicates are moved to the front instead of being added again."""
    history = scene.oshan_recent_vcols
    max_slots = 5

    # Check for duplicate (tolerance-based comparison)
    dup_index = -1
    for i, item in enumerate(history):
        if all(abs(item.color[c] - color_tuple[c]) < 1e-4 for c in range(4)):
            dup_index = i
            break

    if dup_index >= 0:
        # Move existing entry to front
        history.move(dup_index, 0)
    else:
        # Insert a new entry at front
        history.add()
        # The new item is appended at the end; move it to index 0
        history.move(len(history) - 1, 0)
        history[0].color = color_tuple

        # Trim to max_slots
        while len(history) > max_slots:
            history.remove(len(history) - 1)


class NASH3D_OT_pick_recent_color(bpy.types.Operator):
    """Set the active vertex paint fill color from a color swatch"""
    bl_idname = "nash3d.pick_recent_color"
    bl_label = "Pick Swatch Color"
    bl_options = {'INTERNAL'}

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        scene = context.scene
        history = scene.oshan_recent_vcols
        if 0 <= self.index < len(history):
            scene.oshan_vcol_fill_color = history[self.index].color
        return {'FINISHED'}


class NASH3D_OT_add_to_swatch(bpy.types.Operator):
    """Add the current Fill Color to the Color Swatch"""
    bl_idname = "nash3d.add_to_swatch"
    bl_label = "Add to Swatch"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        scene = context.scene
        color = tuple(scene.oshan_vcol_fill_color)
        push_recent_color(scene, color)
        return {'FINISHED'}


class NASH3D_OT_pick_value_color(bpy.types.Operator):
    """Set the active vertex paint fill color from a generated value swatch"""
    bl_idname = "nash3d.pick_value_color"
    bl_label = "Pick Value Swatch Color"
    bl_options = {'INTERNAL'}

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        scene = context.scene
        history = scene.oshan_value_vcols
        if 0 <= self.index < len(history):
            scene.oshan_vcol_fill_color = history[self.index].color
        return {'FINISHED'}


class NASH3D_OT_generate_value_swatch(bpy.types.Operator):
    """Generate a 5-color swatch with varying values based on the current Fill Color"""
    bl_idname = "nash3d.generate_value_swatch"
    bl_label = "Generate Value Swatch"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        scene = context.scene
        history = scene.oshan_value_vcols
        base_color = scene.oshan_vcol_fill_color
        
        import mathutils
        c = mathutils.Color((base_color[0], base_color[1], base_color[2]))
        alpha = base_color[3]
        
        history.clear()
        
        # 5 colors with varying values
        values = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        for v in values:
            c.v = v
            item = history.add()
            item.color = (c.r, c.g, c.b, alpha)
            
        return {'FINISHED'}


# =========================================================================
# CLEAN VERTEX COLORS OPERATOR
# =========================================================================
class NASH3D_OT_clean_vertex_colors(bpy.types.Operator):
    """Fill the vertex colors of all selected mesh objects with a chosen color.
    Creates a vertex color layer if the object does not already have one."""
    bl_idname = "nash3d.clean_vertex_colors"
    bl_label = "Clean Vertex Colors"
    bl_options = {'REGISTER', 'UNDO'}

    # Color property — receives the value from the scene color picker via the panel
    fill_color: bpy.props.FloatVectorProperty(
        name="Fill Color",
        description="Color to fill the vertex color layer with",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0)
    )

    @classmethod
    def poll(cls, context):
        return any(obj.type == 'MESH' for obj in context.selected_objects)

    def execute(self, context):
        processed_objects = 0
        processed_layers = 0
        target_color = tuple(self.fill_color)  # (R, G, B, A)

        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue

            mesh = obj.data

            # --- Blender 3.2+ Color Attributes (preferred modern API) ---
            if hasattr(mesh, 'color_attributes'):
                # Create a new layer if none exists
                if not mesh.color_attributes:
                    mesh.color_attributes.new(
                        name="Col",
                        type='BYTE_COLOR',
                        domain='CORNER'
                    )

                for attr in mesh.color_attributes:
                    for item in attr.data:
                        item.color = target_color
                    processed_layers += 1

            # --- Legacy vertex color layers (Blender < 3.2 / fallback) ---
            else:
                # Create a new layer if none exists
                if not mesh.vertex_colors:
                    mesh.vertex_colors.new(name="Col")

                for vcol_layer in mesh.vertex_colors:
                    for loop_col in vcol_layer.data:
                        loop_col.color = target_color
                    processed_layers += 1

            mesh.update()
            processed_objects += 1

        if processed_objects == 0:
            self.report({'WARNING'}, "No selected mesh objects found.")
            return {'CANCELLED'}

        r, g, b, a = target_color
        # Push the applied color into the recent-colors history
        push_recent_color(context.scene, target_color)

        self.report(
            {'INFO'},
            f"Filled {processed_layers} vertex color layer(s) across {processed_objects} object(s) "
            f"with color ({r:.2f}, {g:.2f}, {b:.2f})."
        )
        return {'FINISHED'}


# =========================================================================
# RANDOM VERTEX COLORS OPERATOR
# =========================================================================
class NASH3D_OT_random_vertex_colors(bpy.types.Operator):
    """Assign a unique random solid color to each selected mesh object's vertex color layer.
    Creates a vertex color layer if the object does not already have one."""
    bl_idname = "nash3d.random_vertex_colors"
    bl_label = "Add Random Vertex Colors"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(obj.type == 'MESH' for obj in context.selected_objects)

    def execute(self, context):
        import random
        processed_objects = 0

        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue

            mesh = obj.data

            # Generate one unique random color for this object (full alpha)
            rand_color = (random.random(), random.random(), random.random(), 1.0)

            # --- Blender 3.2+ Color Attributes (preferred modern API) ---
            if hasattr(mesh, 'color_attributes'):
                # Create a new layer if none exists
                if not mesh.color_attributes:
                    mesh.color_attributes.new(
                        name="Col",
                        type='BYTE_COLOR',
                        domain='CORNER'
                    )

                for attr in mesh.color_attributes:
                    for item in attr.data:
                        item.color = rand_color

            # --- Legacy vertex color layers (Blender < 3.2 / fallback) ---
            else:
                # Create a new layer if none exists
                if not mesh.vertex_colors:
                    mesh.vertex_colors.new(name="Col")

                for vcol_layer in mesh.vertex_colors:
                    for loop_col in vcol_layer.data:
                        loop_col.color = rand_color

            mesh.update()
            processed_objects += 1

        if processed_objects == 0:
            self.report({'WARNING'}, "No selected mesh objects found.")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Assigned random colors to {processed_objects} object(s).")
        return {'FINISHED'}


# =========================================================================
# EASY SMOOTH OPERATOR
# =========================================================================
class NASH3D_OT_easy_smooth(bpy.types.Operator):
    """Clear custom normals, make smooth, mark sharp by angle, and add Weighted Normal modifier"""
    bl_idname = "nash3d.easy_smooth"
    bl_label = "Smooth it!"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        import bmesh
        import math
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Active object must be a Mesh.")
            return {'CANCELLED'}

        original_mode = obj.mode
        
        try:
            is_edit_mode = (original_mode == 'EDIT')
            
            # Sync selection data to obj.data
            if obj.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
                
            process_all = True
            if is_edit_mode:
                if any(e.select for e in obj.data.edges):
                    process_all = False
            
            # 1. Delete custom normal data, sharp edges, and make Shade Smooth
            # Clear Custom Split Normals
            try:
                bpy.ops.mesh.customdata_custom_splitnormals_clear()
            except Exception as e:
                pass
                
            # Clear sharp edges and smooth faces using BMesh
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            
            for edge in bm.edges:
                if process_all or edge.select:
                    edge.smooth = True
                    
            for face in bm.faces:
                if process_all or face.select:
                    face.smooth = True
                    
            bm.to_mesh(obj.data)
            bm.free()
            obj.data.update()
            
            # 2. Make sharp edges by angle
            bpy.ops.object.mode_set(mode='EDIT')
            if process_all:
                bpy.ops.mesh.select_all(action='SELECT')
                
            # Apply set_sharpness_by_angle
            angle_rad = context.scene.oshan_smooth_angle
            bpy.ops.mesh.set_sharpness_by_angle(angle=angle_rad)
            
            # 3. Add/Configure Weighted Normal modifier (must be in OBJECT mode)
            if obj.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
                
            weighted_normal_mods = [m for m in obj.modifiers if m.type == 'WEIGHTED_NORMAL']
            if not weighted_normal_mods:
                mod = obj.modifiers.new(name="WeightedNormal", type='WEIGHTED_NORMAL')
                mod.keep_sharp = True
                bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)
                msg_suffix = "WeightedNormal modifier added at index 0."
            else:
                for i, mod in enumerate(weighted_normal_mods):
                    mod.keep_sharp = True
                    if i == 0:
                        bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)
                msg_suffix = "WeightedNormal modifier configured and moved to index 0."
                
            # Restore the original mode
            if original_mode != obj.mode:
                bpy.ops.object.mode_set(mode=original_mode)
                
            deg = round(math.degrees(angle_rad))
            self.report({'INFO'}, f"Applied Easy Smooth: custom normals cleared, sharp by {deg}° applied, and {msg_suffix}")
            return {'FINISHED'}
            
        except Exception as e:
            # Safely restore mode in case of failure
            if obj.mode != original_mode:
                try:
                    bpy.ops.object.mode_set(mode=original_mode)
                except:
                    pass
            self.report({'ERROR'}, f"Failed to apply Easy Smooth: {str(e)}")
            return {'CANCELLED'}


# =========================================================================
# UPDATER OPERATOR
# =========================================================================
class NASH3D_OT_update_addon(bpy.types.Operator):
    """Overwrites the current addon file with the selected Python file"""
    bl_idname = "nash3d.update_addon"
    bl_label = "Apply Update"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        preferences = get_preferences(context)
        if not preferences:
            self.report({'ERROR'}, "Addon preferences not found. Ensure the addon is installed and enabled.")
            return {'CANCELLED'}

        new_file_path = bpy.path.abspath(preferences.update_file_path)

        if not new_file_path:
            self.report({'ERROR'}, "No file selected. Please choose a .py file.")
            return {'CANCELLED'}

        if not os.path.exists(new_file_path):
            self.report({'ERROR'}, f"Selected file does not exist: {new_file_path}")
            return {'CANCELLED'}

        if not new_file_path.endswith(".py"):
            self.report({'ERROR'}, "Selected file must be a Python (.py) file.")
            return {'CANCELLED'}

        current_file_path = os.path.realpath(__file__)

        # Prevent overwriting the file with itself
        if os.path.samefile(new_file_path, current_file_path):
            self.report({'ERROR'}, "Selected file is already the active addon file.")
            return {'CANCELLED'}

        backup_path = current_file_path + ".bak"

        try:
            # Create a backup copy in case copy fails
            shutil.copyfile(current_file_path, backup_path)
            # Overwrite the addon file with the new file
            shutil.copyfile(new_file_path, current_file_path)
            
            # Remove backup on success
            if os.path.exists(backup_path):
                os.remove(backup_path)

            self.report({'INFO'}, "Addon updated successfully! Reloading scripts...")
            
            # Reload all Blender scripts/addons to load the updated file immediately
            bpy.ops.script.reload()
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to update addon: {str(e)}")
            # Restore original file if backup exists
            if os.path.exists(backup_path):
                try:
                    shutil.copyfile(backup_path, current_file_path)
                    os.remove(backup_path)
                except:
                    pass
            return {'CANCELLED'}


# =========================================================================
# ADDON PREFERENCES
# =========================================================================
class Nash3D_QuickPanel_Preferences(bpy.types.AddonPreferences):
    """Preferences panel displayed in Edit > Preferences > Addons"""
    bl_idname = __package__ or __name__

    update_file_path: bpy.props.StringProperty(
        name="Update File",
        description="Select the updated .py file for this addon",
        subtype='FILE_PATH',
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Update Addon from Local File:")
        
        row = layout.row(align=True)
        row.prop(self, "update_file_path", text="")
        row.operator("nash3d.update_addon", text="Apply Update", icon='FILE_REFRESH')


# =========================================================================
# UI PANEL (PARENT CONTAINER)
# =========================================================================
class VIEW3D_PT_oshan_quick_tools(bpy.types.Panel):
    """Parent Panel in the 3D Viewport Sidebar (N-panel)"""
    bl_label = "Quick Tools"
    bl_idname = "VIEW3D_PT_oshan_quick_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Oshan Tools'  # Category/tab name in the N-panel
    # bl_context = 'objectmode'    # Commented out to display panel in Edit Mode for Easy Bevel

    def draw(self, context):
        # Container panel draws nothing directly, acting as a clean wrapper for collapsible subpanels
        pass


# =========================================================================
# COLLAPSIBLE SUBPANEL (LATTICE DEFORMER)
# =========================================================================
class VIEW3D_PT_oshan_lattice_subpanel(bpy.types.Panel):
    """Collapsible Subpanel for the Lattice Deformer tool"""
    bl_label = "Lattice Deformer"
    bl_idname = "VIEW3D_PT_oshan_lattice_subpanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Oshan Tools'
    bl_parent_id = "VIEW3D_PT_oshan_quick_tools"  # Declares it as a subpanel
    bl_options = {'DEFAULT_CLOSED'}               # Starts collapsed

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        col = layout.column(align=True)
        
        # Row for U Axis interpolation and resolution
        row = col.row(align=True)
        row.label(text="U Axis:")
        row.prop(scene, "oshan_lattice_u", text="")
        row.prop(scene, "oshan_lattice_res_u", text="Res")
        
        # Row for V Axis interpolation and resolution
        row = col.row(align=True)
        row.label(text="V Axis:")
        row.prop(scene, "oshan_lattice_v", text="")
        row.prop(scene, "oshan_lattice_res_v", text="Res")
        
        # Row for W Axis interpolation and resolution
        row = col.row(align=True)
        row.label(text="W Axis:")
        row.prop(scene, "oshan_lattice_w", text="")
        row.prop(scene, "oshan_lattice_res_w", text="Res")
        
        layout.separator()
        
        # Operator button passing N-panel values to operator properties
        op = layout.operator("nash3d.create_lattice", text="Create Lattice Deformer", icon='ADD')
        op.interpolation_u = scene.oshan_lattice_u
        op.interpolation_v = scene.oshan_lattice_v
        op.interpolation_w = scene.oshan_lattice_w
        op.resolution_u = scene.oshan_lattice_res_u
        op.resolution_v = scene.oshan_lattice_res_v
        op.resolution_w = scene.oshan_lattice_res_w


# =========================================================================
# COLLAPSIBLE SUBPANEL (GROUP TOGETHER)
# =========================================================================
class VIEW3D_PT_oshan_group_subpanel(bpy.types.Panel):
    """Collapsible Subpanel for Grouping objects together"""
    bl_label = "Group Together"
    bl_idname = "VIEW3D_PT_oshan_group_subpanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Oshan Tools'
    bl_parent_id = "VIEW3D_PT_oshan_quick_tools"  # Declares it as a subpanel
    bl_options = {'DEFAULT_CLOSED'}               # Starts collapsed

    def draw(self, context):
        layout = self.layout
        layout.operator("nash3d.group_together", text="Group", icon='OUTLINER_OB_EMPTY')


# =========================================================================
# COLLAPSIBLE SUBPANEL (EASY BEVEL)
# =========================================================================
class VIEW3D_PT_oshan_easy_bevel_subpanel(bpy.types.Panel):
    """Collapsible Subpanel for the Easy Bevel tool"""
    bl_label = "Easy Bevel"
    bl_idname = "VIEW3D_PT_oshan_easy_bevel_subpanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Oshan Tools'
    bl_parent_id = "VIEW3D_PT_oshan_quick_tools"  # Declares it as a subpanel
    bl_options = {'DEFAULT_CLOSED'}               # Starts collapsed

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.operator("nash3d.easy_bevel", text="Bevel it!", icon='MOD_BEVEL')
        row.operator("nash3d.unbevel", text="Unbevel It!", icon='X')


# =========================================================================
# COLLAPSIBLE SUBPANEL (EASY SMOOTH)
# =========================================================================
class VIEW3D_PT_oshan_easy_smooth_subpanel(bpy.types.Panel):
    """Collapsible Subpanel for the Easy Smooth tool"""
    bl_label = "Easy Smooth"
    bl_idname = "VIEW3D_PT_oshan_easy_smooth_subpanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Oshan Tools'
    bl_parent_id = "VIEW3D_PT_oshan_quick_tools"  # Declares it as a subpanel
    bl_options = {'DEFAULT_CLOSED'}               # Starts collapsed

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        col = layout.column(align=True)
        col.prop(scene, "oshan_smooth_angle", text="Angle")
        col.separator(factor=0.5)
        col.operator("nash3d.easy_smooth", text="Smooth it!", icon='MOD_SMOOTH')
        
        col.separator(factor=0.5)
        note = col.column(align=True)
        note.scale_y = 0.8
        note.label(text="Smooths & sharpens edges by angle.")
        note.label(text="Adds Weighted Normal modifier.")
        note.label(text="*Edit Mode: Affects selected edges only.")


# =========================================================================
# COLLAPSIBLE SUBPANEL (VERTEX PAINT TOOLS)
# =========================================================================
class VIEW3D_PT_oshan_vertex_paint_subpanel(bpy.types.Panel):
    """Collapsible Subpanel for Vertex Paint utilities"""
    bl_label = "Vertex Paint Tools"
    bl_idname = "VIEW3D_PT_oshan_vertex_paint_subpanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Oshan Tools'
    bl_parent_id = "VIEW3D_PT_oshan_quick_tools"  # Declares it as a subpanel
    bl_options = {'DEFAULT_CLOSED'}               # Starts collapsed
    bl_context = 'objectmode'                     # Only visible in Object Mode

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        col = layout.column(align=True)
        # Color picker swatch
        col.prop(scene, "oshan_vcol_fill_color", text="Fill Color")
        col.separator(factor=0.5)
        # Button — passes the scene color into the operator property
        op = col.operator("nash3d.clean_vertex_colors", text="Apply Vertex Color", icon='BRUSH_DATA')
        op.fill_color = scene.oshan_vcol_fill_color
        col.separator(factor=0.5)
        # Button — assigns a unique random color to each selected object
        col.operator("nash3d.random_vertex_colors", text="Add Random Vertex Colors", icon='COLOR')

        # ---- Color Swatch ----
        col.separator(factor=1.0)
        col.label(text="Recent Colors:")
        history = scene.oshan_recent_vcols
        if len(history) > 0:
            # Color swatches row
            swatch_row = col.row(align=True)
            for i in range(len(history)):
                swatch_row.prop(history[i], "color", text="")
            # Pick buttons row — click to load that color as the active Fill Color
            pick_row = col.row(align=True)
            for i in range(len(history)):
                op = pick_row.operator("nash3d.pick_recent_color", text="Pick")
                op.index = i
            col.separator(factor=0.5)
        
        # Always show Add to Swatch button
        col.operator("nash3d.add_to_swatch", text="Add to Swatch", icon='ADD')
        
        # ---- Generated Value Swatch ----
        col.separator(factor=1.0)
        col.label(text="Value Swatch:")
        value_history = scene.oshan_value_vcols
        if len(value_history) > 0:
            # Color swatches row
            vswatch_row = col.row(align=True)
            for i in range(len(value_history)):
                vswatch_row.prop(value_history[i], "color", text="")
            # Pick buttons row
            vpick_row = col.row(align=True)
            for i in range(len(value_history)):
                op = vpick_row.operator("nash3d.pick_value_color", text="Pick")
                op.index = i
            col.separator(factor=0.5)
            
        col.operator("nash3d.generate_value_swatch", text="Generate Value Swatch", icon='COLORSET_13_VEC')


# =========================================================================
# ABOUT SUBPANEL
# =========================================================================
class VIEW3D_PT_oshan_about_subpanel(bpy.types.Panel):
    """About subpanel — always expanded at the bottom of the Quick Tools panel"""
    bl_label = "About"
    bl_idname = "VIEW3D_PT_oshan_about_subpanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Oshan Tools'
    bl_parent_id = "VIEW3D_PT_oshan_quick_tools"  # Declares it as a subpanel
    # No 'DEFAULT_CLOSED' — panel starts expanded so the text is always visible

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Created by Oshan Devinda", icon='USER')
        col.label(text="(aka Nasho3D)")


# =========================================================================
# UV TOOLS PANEL (UV EDITOR SIDEBAR)
# =========================================================================
class IMAGE_PT_oshan_uv_tools(bpy.types.Panel):
    """Panel in the UV Editor Sidebar (N-panel)"""
    bl_label = "UV Tools"
    bl_idname = "IMAGE_PT_oshan_uv_tools"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Oshan Tools'

    @classmethod
    def poll(cls, context):
        return (context.area and 
                context.area.type == 'IMAGE_EDITOR' and 
                context.area.ui_type == 'UV')

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        if not (obj and obj.type == 'MESH' and obj.mode == 'EDIT'):
            layout.label(text="Enter Edit Mode with a Mesh", icon='INFO')


# =========================================================================
# SNAPPING TOOLS SUBPANEL
# =========================================================================
class IMAGE_PT_oshan_uv_snapping(bpy.types.Panel):
    bl_label = "Snapping tools"
    bl_idname = "IMAGE_PT_oshan_uv_snapping"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Oshan Tools'
    bl_parent_id = "IMAGE_PT_oshan_uv_tools"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (context.area and 
                context.area.type == 'IMAGE_EDITOR' and 
                context.area.ui_type == 'UV')

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        scene = context.scene
        
        if not (obj and obj.type == 'MESH' and obj.mode == 'EDIT'):
            return
            
        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(scene, "oshan_uv_snap_u", text="U")
        row.prop(scene, "oshan_uv_snap_v", text="V")
        
        op = col.operator("nash3d.snap_to_coordinate", text="Snap to coordinate", icon='UV_SYNC_SELECT')
        op.target_u = scene.oshan_uv_snap_u
        op.target_v = scene.oshan_uv_snap_v
        
        col.separator(factor=0.5)
        col.operator("nash3d.snap_to_vertex", text="Snap to Vertex", icon='SNAP_VERTEX')


# =========================================================================
# ROTATING TOOLS SUBPANEL
# =========================================================================
class IMAGE_PT_oshan_uv_rotating(bpy.types.Panel):
    bl_label = "Rotating tools"
    bl_idname = "IMAGE_PT_oshan_uv_rotating"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Oshan Tools'
    bl_parent_id = "IMAGE_PT_oshan_uv_tools"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (context.area and 
                context.area.type == 'IMAGE_EDITOR' and 
                context.area.ui_type == 'UV')

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        scene = context.scene
        
        if not (obj and obj.type == 'MESH' and obj.mode == 'EDIT'):
            return
            
        col = layout.column(align=True)
        col.prop(scene, "oshan_uv_rotate_pivot", text="Pivot")
        col.prop(scene, "oshan_uv_rotate_increment", text="Rotation (°)")
        
        row2 = col.row(align=True)
        row2.scale_y = 1.0
        
        op_ccw = row2.operator("nash3d.rotate_uv_island", text="Rotate Left", icon='LOOP_BACK')
        op_ccw.angle = scene.oshan_uv_rotate_increment
        op_ccw.pivot = scene.oshan_uv_rotate_pivot
        
        op_cw = row2.operator("nash3d.rotate_uv_island", text="Rotate Right", icon='LOOP_FORWARDS')
        op_cw.angle = -scene.oshan_uv_rotate_increment
        op_cw.pivot = scene.oshan_uv_rotate_pivot


# =========================================================================
# TEXEL DENSITY SUBPANEL (UV EDITOR SIDEBAR)
# =========================================================================
class IMAGE_PT_oshan_texel_density(bpy.types.Panel):
    """Subpanel for the Texel Density tools in the UV Editor Sidebar"""
    bl_label = "Texel Density"
    bl_idname = "IMAGE_PT_oshan_texel_density"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Oshan Tools'
    bl_parent_id = "IMAGE_PT_oshan_uv_tools"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (context.area and
                context.area.type == 'IMAGE_EDITOR' and
                context.area.ui_type == 'UV')

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        scene = context.scene

        if not (obj and obj.type == 'MESH' and obj.mode == 'EDIT'):
            layout.label(text="Enter Edit Mode with a Mesh", icon='INFO')
            return

        col = layout.column(align=True)
        col.prop(scene, "oshan_td_texture_size", text="Texture Size")
        col.prop(scene, "oshan_td_physical_size", text="Physical Size (m)")
        
        col.separator(factor=0.5)
        col.prop(scene, "oshan_td_scale_pivot", text="Pivot")
        op = col.operator("nash3d.set_texel_density", text="Set Texel Density", icon='UV_DATA')
        op.pivot = scene.oshan_td_scale_pivot


# =========================================================================
# REGISTRATION
# =========================================================================
# Register/unregister classes in this tuple to make them active in Blender
classes = (
    NASH3D_RecentColorItem,
    NASH3D_OT_pick_recent_color,
    NASH3D_OT_pick_value_color,
    NASH3D_OT_add_to_swatch,
    NASH3D_OT_generate_value_swatch,
    NASH3D_OT_snap_to_coordinate,
    NASH3D_OT_snap_to_vertex,
    NASH3D_OT_rotate_uv_island,
    NASH3D_OT_set_texel_density,
    NASH3D_OT_create_lattice,
    NASH3D_OT_group_together,
    NASH3D_OT_easy_bevel,
    NASH3D_OT_unbevel,
    NASH3D_OT_easy_smooth,
    NASH3D_OT_clean_vertex_colors,
    NASH3D_OT_random_vertex_colors,
    NASH3D_OT_update_addon,
    Nash3D_QuickPanel_Preferences,
    VIEW3D_PT_oshan_quick_tools,
    VIEW3D_PT_oshan_lattice_subpanel,
    VIEW3D_PT_oshan_group_subpanel,
    VIEW3D_PT_oshan_easy_bevel_subpanel,
    VIEW3D_PT_oshan_easy_smooth_subpanel,
    VIEW3D_PT_oshan_vertex_paint_subpanel,
    VIEW3D_PT_oshan_about_subpanel,
    IMAGE_PT_oshan_uv_tools,
    IMAGE_PT_oshan_uv_snapping,
    IMAGE_PT_oshan_uv_rotating,
    IMAGE_PT_oshan_texel_density,
    # OBJECT_OT_my_custom_operator, # Add/uncomment future operators here
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Register easy smooth angle
    bpy.types.Scene.oshan_smooth_angle = bpy.props.FloatProperty(
        name="Sharpness Angle",
        description="Angle threshold for making edges sharp",
        subtype='ANGLE',
        default=0.523599, # 30 degrees in radians
        min=0.0,
        max=3.14159
    )

    # Register vertex paint fill color
    bpy.types.Scene.oshan_vcol_fill_color = bpy.props.FloatVectorProperty(
        name="Fill Color",
        description="Color used to fill vertex color layers on selected objects",
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 1.0)
    )

    # Register recent vertex colors collection
    bpy.types.Scene.oshan_recent_vcols = bpy.props.CollectionProperty(
        type=NASH3D_RecentColorItem,
        name="Recent Vertex Colors",
        description="Last 5 vertex colors used with Apply Vertex Color"
    )

    # Register generated value swatch collection
    bpy.types.Scene.oshan_value_vcols = bpy.props.CollectionProperty(
        type=NASH3D_RecentColorItem,
        name="Value Vertex Colors",
        description="5 generated vertex colors with varying value"
    )

    # Register texel density properties
    bpy.types.Scene.oshan_td_texture_size = bpy.props.EnumProperty(
        name="Texture Size",
        description="Texture resolution that occupies the full 0–1 UV space",
        items=[
            ('128',  "128 px",  "128 × 128 pixels"),
            ('256',  "256 px",  "256 × 256 pixels"),
            ('512',  "512 px",  "512 × 512 pixels"),
            ('1024', "1024 px", "1024 × 1024 pixels"),
            ('2048', "2048 px", "2048 × 2048 pixels"),
            ('4096', "4096 px", "4096 × 4096 pixels"),
            ('8192', "8192 px", "8192 × 8192 pixels"),
        ],
        default='2048'
    )
    bpy.types.Scene.oshan_td_physical_size = bpy.props.FloatProperty(
        name="Physical Size",
        description="Real-world size (in metres) that the full texture covers",
        default=1.0,
        min=0.001,
        soft_max=100.0,
        unit='LENGTH'
    )
    bpy.types.Scene.oshan_td_scale_pivot = bpy.props.EnumProperty(
        name="Pivot Point",
        description="Pivot point for scaling",
        items=[
            ('BBOX_CENTER', "BBox Center", "Scale around the bounding box center of the island"),
            ('VERTEX', "Vertex", "Scale around the selected vertex (midpoint if multiple)"),
        ],
        default='BBOX_CENTER'
    )

    # Register snap coordinates for UV tools
    bpy.types.Scene.oshan_uv_snap_u = bpy.props.FloatProperty(
        name="Target U",
        description="Target U coordinate to snap the UV island to",
        default=0.0
    )
    bpy.types.Scene.oshan_uv_snap_v = bpy.props.FloatProperty(
        name="Target V",
        description="Target V coordinate to snap the UV island to",
        default=1.0
    )
    bpy.types.Scene.oshan_uv_rotate_increment = bpy.props.FloatProperty(
        name="Rotation Increment",
        description="Rotation step in degrees for the CW/CCW buttons",
        default=90.0,
        min=0.1,
        max=180.0
    )
    bpy.types.Scene.oshan_uv_rotate_pivot = bpy.props.EnumProperty(
        name="Pivot Point",
        description="Pivot point for rotation",
        items=[
            ('ISLAND_CENTER', "Island Center", "Rotate around the bounding box center of the island"),
            ('BBOX_CENTER', "BBox Center", "Rotate around the selection's bounding box center"),
            ('VERTEX', "Vertex", "Rotate around the selected vertex (midpoint if multiple)"),
        ],
        default='VERTEX'
    )

    # Register persistent Scene properties for UI dropdowns
    bpy.types.Scene.oshan_lattice_u = bpy.props.EnumProperty(
        name="U Axis Interpolation",
        items=[
            ('KEY_BSPLINE', "B-Spline", "B-Spline interpolation"),
            ('KEY_LINEAR', "Linear", "Linear interpolation"),
            ('KEY_CARDINAL', "Cardinal", "Cardinal interpolation"),
            ('KEY_CATMULL_ROM', "Catmull-Rom", "Catmull-Rom interpolation"),
        ],
        default='KEY_LINEAR'
    )
    bpy.types.Scene.oshan_lattice_v = bpy.props.EnumProperty(
        name="V Axis Interpolation",
        items=[
            ('KEY_BSPLINE', "B-Spline", "B-Spline interpolation"),
            ('KEY_LINEAR', "Linear", "Linear interpolation"),
            ('KEY_CARDINAL', "Cardinal", "Cardinal interpolation"),
            ('KEY_CATMULL_ROM', "Catmull-Rom", "Catmull-Rom interpolation"),
        ],
        default='KEY_LINEAR'
    )
    bpy.types.Scene.oshan_lattice_w = bpy.props.EnumProperty(
        name="W Axis Interpolation",
        items=[
            ('KEY_BSPLINE', "B-Spline", "B-Spline interpolation"),
            ('KEY_LINEAR', "Linear", "Linear interpolation"),
            ('KEY_CARDINAL', "Cardinal", "Cardinal interpolation"),
            ('KEY_CATMULL_ROM', "Catmull-Rom", "Catmull-Rom interpolation"),
        ],
        default='KEY_LINEAR'
    )

    # Register persistent Scene properties for UI resolutions
    bpy.types.Scene.oshan_lattice_res_u = bpy.props.IntProperty(
        name="U Axis Resolution",
        description="Number of control points along the U axis",
        default=2,
        min=2,
        max=64
    )
    bpy.types.Scene.oshan_lattice_res_v = bpy.props.IntProperty(
        name="V Axis Resolution",
        description="Number of control points along the V axis",
        default=2,
        min=2,
        max=64
    )
    bpy.types.Scene.oshan_lattice_res_w = bpy.props.IntProperty(
        name="W Axis Resolution",
        description="Number of control points along the W axis",
        default=2,
        min=2,
        max=64
    )

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    # Delete easy smooth angle
    del bpy.types.Scene.oshan_smooth_angle

    # Delete vertex paint fill color and recent colors
    del bpy.types.Scene.oshan_value_vcols
    del bpy.types.Scene.oshan_recent_vcols
    del bpy.types.Scene.oshan_vcol_fill_color

    # Delete texel density properties
    del bpy.types.Scene.oshan_td_texture_size
    del bpy.types.Scene.oshan_td_physical_size
    del bpy.types.Scene.oshan_td_scale_pivot

    # Delete snap coordinates and rotation increment for UV tools
    del bpy.types.Scene.oshan_uv_snap_u
    del bpy.types.Scene.oshan_uv_snap_v
    del bpy.types.Scene.oshan_uv_rotate_increment
    del bpy.types.Scene.oshan_uv_rotate_pivot

    # Delete persistent Scene properties
    del bpy.types.Scene.oshan_lattice_u
    del bpy.types.Scene.oshan_lattice_v
    del bpy.types.Scene.oshan_lattice_w
    del bpy.types.Scene.oshan_lattice_res_u
    del bpy.types.Scene.oshan_lattice_res_v
    del bpy.types.Scene.oshan_lattice_res_w

if __name__ == "__main__":
    register()
