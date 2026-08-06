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
import bmesh
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
    """Move the selected UV island to place the selected vertex at the target coordinates"""
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
            
            # Calculate translation needed to move the selected vertex to the target coordinates
            translation_u = self.target_u - u_sel
            translation_v = self.target_v - v_sel
            
            # Apply the translation directly to loops
            for l in island:
                l[uv_layer].uv.x += translation_u
                l[uv_layer].uv.y += translation_v
                    
            processed_count += 1
            
        if processed_count > 0:
            bmesh.update_edit_mesh(obj.data)
            self.report({'INFO'}, f"Snapped {processed_count} island(s) to ({self.target_u:.2f}, {self.target_v:.2f}).")
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
        if not active_face:
            self.report({'ERROR'}, "Please ensure there is an active face in the target island.")
            return {'CANCELLED'}
            
        islands = get_uv_islands(bm, uv_layer)
        
        active_island = None
        if active_face.loops:
            first_loop = active_face.loops[0]
            for island in islands:
                if first_loop in island:
                    active_island = island
                    break
                    
        if not active_island:
            self.report({'ERROR'}, "Could not find the island containing the active face.")
            return {'CANCELLED'}
            
        target_group = None
        source_group = None
        
        for grp in uv_groups:
            if grp[0] in active_island:
                target_group = grp
            else:
                source_group = grp
                
        if not target_group:
            self.report({'ERROR'}, "No selected vertex found in the active face's island (Target Island).")
            return {'CANCELLED'}
            
        if not source_group:
            self.report({'ERROR'}, "No selected vertex found in the secondary island (Source Island).")
            return {'CANCELLED'}
            
        target_island = active_island
        source_island = None
        for island in islands:
            if source_group[0] in island:
                source_island = island
                break
                
        if not source_island:
            self.report({'ERROR'}, "Could not determine the island for the second selected vertex.")
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
# GET TEXEL DENSITY OPERATOR
# =========================================================================
class NASH3D_OT_get_texel_density(bpy.types.Operator):
    """Read the texel density of the selected UV island(s) and update the
    Physical Size field so the TD display reflects the actual island density.
    Uses an area-weighted method across all faces for accuracy — this is more
    correct than a simple per-face average when faces vary in size.
    Formula: Physical Size = Texture Size / Current TD"""
    bl_idname = "nash3d.get_texel_density"
    bl_label = "Get Texel Density"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.active_object and
                context.active_object.type == 'MESH' and
                context.active_object.mode == 'EDIT')

    # ---- private helpers (shared with set_texel_density) ----------------

    @staticmethod
    def _world_face_area(face, matrix_world):
        """Return the world-space area of a BMFace via fan triangulation."""
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
        import math

        scene = context.scene
        texture_size = int(scene.oshan_td_texture_size)  # pixels

        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active

        if not uv_layer:
            self.report({'ERROR'}, "No active UV layer found.")
            return {'CANCELLED'}

        matrix_world = obj.matrix_world
        islands = get_uv_islands(bm, uv_layer)

        # Accumulate areas across ALL selected islands for a combined result
        total_world_area = 0.0
        total_uv_area = 0.0
        processed = 0

        for island in islands:
            # Only process islands that have at least one selected loop/vert
            sel_loops = [
                l for l in island
                if (l.uv_select_vert if hasattr(l, "uv_select_vert") else l[uv_layer].select)
            ]
            if not sel_loops:
                continue

            # Group loops by face
            face_loop_map = {}
            for l in island:
                fid = l.face.index
                if fid not in face_loop_map:
                    face_loop_map[fid] = []
                face_loop_map[fid].append(l)

            # Accumulate world and UV areas for this island
            for loops in face_loop_map.values():
                total_world_area += self._world_face_area(loops[0].face, matrix_world)
                total_uv_area    += self._uv_face_area(loops, uv_layer)

            processed += 1

        if processed == 0:
            self.report({'WARNING'}, "No selected UV islands found.")
            return {'CANCELLED'}

        if total_world_area < 1e-12 or total_uv_area < 1e-12:
            self.report({'ERROR'}, "Island has zero area — cannot calculate texel density.")
            return {'CANCELLED'}

        # current_td  =  texture_size * sqrt(uv_area / world_area)  [texels/m]
        # physical_size  =  texture_size / current_td
        current_td = texture_size * math.sqrt(total_uv_area / total_world_area)
        physical_size = texture_size / current_td

        # Write the back-calculated Physical Size back to the scene property
        scene.oshan_td_physical_size = physical_size

        self.report(
            {'INFO'},
            f"Got texel density: {current_td:.2f} px/m — "
            f"Physical Size set to {physical_size:.4f} m "
            f"(from {processed} island(s), {texture_size}px texture)."
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
# =========================================================================
# EXPLODE ISLANDS OPERATOR
# =========================================================================
class NASH3D_OT_explode_islands(bpy.types.Operator):
    """Spread all selected UV islands apart with a margin so they are clearly visible and easy to arrange manually"""
    bl_idname = "nash3d.explode_islands"
    bl_label = "Explode Islands"
    bl_options = {'REGISTER', 'UNDO'}

    margin: bpy.props.FloatProperty(
        name="Margin",
        description="Spacing between islands after exploding",
        default=0.05,
        min=0.0,
        max=1.0
    )

    @classmethod
    def poll(cls, context):
        return (context.active_object and
                context.active_object.type == 'MESH' and
                context.active_object.mode == 'EDIT')

    def execute(self, context):
        obj = context.active_object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.verify()

        # ---- Collect islands using only selected faces ----
        # Each island is a set of loops whose UV positions we will translate
        selected_faces = [f for f in bm.faces if f.select]
        if not selected_faces:
            self.report({'WARNING'}, "No faces selected.")
            return {'CANCELLED'}

        # Build a union-find structure over loops to detect connected UV islands
        loops_in_selected = []
        for f in selected_faces:
            for l in f.loops:
                loops_in_selected.append(l)

        parent = {l: l for l in loops_in_selected}

        def find(l):
            while parent[l] is not l:
                parent[l] = parent[parent[l]]
                l = parent[l]
            return l

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra is not rb:
                parent[ra] = rb

        # Connect loops within the same face
        for f in selected_faces:
            fl = f.loops[:]
            for i in range(1, len(fl)):
                union(fl[0], fl[i])

        # Connect loops that share a vertex AND the same UV coordinate (i.e. are welded in UV space)
        from collections import defaultdict
        vert_to_loops = defaultdict(list)
        for l in loops_in_selected:
            vert_to_loops[l.vert].append(l)

        for vert, vloops in vert_to_loops.items():
            # Group by UV coordinate
            uv_groups = []
            for l in vloops:
                uv = l[uv_layer].uv.copy()
                placed = False
                for grp in uv_groups:
                    if (uv - grp[0][uv_layer].uv).length < 1e-5:
                        grp.append(l)
                        placed = True
                        break
                if not placed:
                    uv_groups.append([l])
            for grp in uv_groups:
                for i in range(1, len(grp)):
                    union(grp[0], grp[i])

        # Collect islands as list of loops grouped by root
        island_map = defaultdict(list)
        for l in loops_in_selected:
            island_map[find(l)].append(l)
        islands = list(island_map.values())

        if len(islands) <= 1:
            self.report({'INFO'}, "Only one island found — nothing to explode.")
            return {'CANCELLED'}

        # ---- Compute AABB for each island ----
        def island_bbox(loops):
            xs = [l[uv_layer].uv.x for l in loops]
            ys = [l[uv_layer].uv.y for l in loops]
            return min(xs), min(ys), max(xs), max(ys)

        bboxes = [island_bbox(isl) for isl in islands]

        # ---- Determine grid layout ----
        import math
        count = len(islands)
        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)

        # Island dimensions (max width/height across all islands for uniform grid)
        widths  = [b[2] - b[0] for b in bboxes]
        heights = [b[3] - b[1] for b in bboxes]
        cell_w  = max(widths)  + self.margin
        cell_h  = max(heights) + self.margin

        # ---- Translate each island so they form a neat grid ----
        for idx, (isl, bb) in enumerate(zip(islands, bboxes)):
            col_i = idx % cols
            row_i = idx // cols

            # Target origin (lower-left corner of the cell)
            target_x = col_i * cell_w
            target_y = row_i * cell_h

            # Current lower-left corner
            cur_x, cur_y = bb[0], bb[1]

            dx = target_x - cur_x
            dy = target_y - cur_y

            for l in isl:
                l[uv_layer].uv.x += dx
                l[uv_layer].uv.y += dy

        bmesh.update_edit_mesh(me)
        self.report({'INFO'}, f"Exploded {len(islands)} UV islands.")
        return {'FINISHED'}


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
        col.separator(factor=0.5)
        col.operator("nash3d.explode_islands", text="Explode Islands", icon='STICKY_UVS_DISABLE')

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
        col.operator("nash3d.get_texel_density", text="Get Texel Density", icon='EYEDROPPER')
        col.separator(factor=0.5)
        col.prop(scene, "oshan_td_scale_pivot", text="Pivot")
        op = col.operator("nash3d.set_texel_density", text="Set Texel Density", icon='UV_DATA')
        op.pivot = scene.oshan_td_scale_pivot

# =========================================================================
# MISC TOOLS OPERATORS & SUBPANEL
# =========================================================================
class NASH3D_OT_remove_all_materials(bpy.types.Operator):
    """Remove materials from all selected objects"""
    bl_idname = "nash3d.remove_all_materials"
    bl_label = "Remove All Materials"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bool(context.selected_objects)

    def execute(self, context):
        count = 0
        for obj in context.selected_objects:
            if hasattr(obj.data, "materials") and obj.data.materials:
                obj.data.materials.clear()
                count += 1
        self.report({'INFO'}, f"Removed materials from {count} objects.")
        return {'FINISHED'}

class NASH3D_OT_copy_materials(bpy.types.Operator):
    """Copy materials from active object to selected objects"""
    bl_idname = "nash3d.copy_materials"
    bl_label = "Copy Materials to Selected"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.selected_objects) > 1 and context.active_object and context.active_object in context.selected_objects

    def execute(self, context):
        active_obj = context.active_object
        selected_objects = context.selected_objects

        if len(selected_objects) <= 1:
            self.report({'WARNING'}, "Select more than one object.")
            return {'CANCELLED'}

        active_materials = active_obj.data.materials if hasattr(active_obj.data, "materials") else []

        count = 0
        for obj in selected_objects:
            if obj != active_obj and hasattr(obj.data, "materials"):
                obj.data.materials.clear()
                for mat in active_materials:
                    obj.data.materials.append(mat)
                count += 1

        self.report({'INFO'}, f"Copied materials to {count} objects.")
        return {'FINISHED'}

class VIEW3D_PT_oshan_misc_tools_subpanel(bpy.types.Panel):
    """Subpanel for Miscellaneous Tools"""
    bl_label = "Misc Tools"
    bl_idname = "VIEW3D_PT_oshan_misc_tools_subpanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Oshan Tools'
    bl_parent_id = "VIEW3D_PT_oshan_quick_tools"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("nash3d.remove_all_materials", text="Remove All Materials", icon='X')
        col.operator("nash3d.copy_materials", text="Copy Materials to Selected", icon='COPYDOWN')

# =========================================================================
# DEFORMING TOOLS OPERATORS & SUBPANEL
# =========================================================================
# =========================================================================
# UV SQUARES HELPER FUNCTIONS
# =========================================================================
from collections import defaultdict
from math import radians, hypot
from timeit import default_timer as timer

precision = 3

BLENDER_5_0_OR_NEWER = bpy.app.version >= (5, 0, 0)

def is_uv_vert_selected(loop, uv_layer):
    if BLENDER_5_0_OR_NEWER:
        return loop.uv_select_vert
    else:
        return loop[uv_layer].select

def set_uv_vert_selected(loop, uv_layer, selected):
    if BLENDER_5_0_OR_NEWER:
        loop.uv_select_vert = selected
    else:
        loop[uv_layer].select = selected

#todo: make joining radius scale with editor zoom rate or average unit length
#todo: align to axis by respect to vert distance
#todo: snap 2dCursor to closest selected vert (when more vertices are selected
#todo: rip different vertex on each press

def main(context, operator, square = False, snapToClosest = False):
    if context.scene.tool_settings.use_uv_select_sync:
        operator.report({'ERROR'}, "Please disable 'Keep UV and edit mesh in sync'")
        # context.scene.tool_settings.use_uv_select_sync = False
        return

    selected_objects = context.selected_objects
    if (context.edit_object not in selected_objects):
        selected_objects.append(context.edit_object)

    for obj in selected_objects:
        if (obj.type == "MESH"):
            main1(obj, context, operator, square, snapToClosest)

def main1(obj, context, operator, square, snapToClosest):
    if context.scene.tool_settings.use_uv_select_sync:
        operator.report({'ERROR'}, "Please disable 'Keep UV and edit mesh in sync'")
        # context.scene.tool_settings.use_uv_select_sync = False
        return

    startTime = timer()
    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    uv_layer = bm.loops.layers.uv.verify()
    # bm.faces.layers.tex.verify()  # currently blender needs both layers.

    edgeVerts, filteredVerts, selFaces, nonQuadFaces, vertsDict, noEdge = ListsOfVerts(uv_layer, bm)

    if len(filteredVerts) == 0: return
    if len(filteredVerts) == 1:
        SnapCursorToClosestSelected(filteredVerts)
        return

    cursorClosestTo = CursorClosestTo(filteredVerts)
    #line is selected

    if len(selFaces) == 0:
        if snapToClosest == True:
            SnapCursorToClosestSelected(filteredVerts)
            return

        VertsDictForLine(uv_layer, bm, filteredVerts, vertsDict)

        if AreVectsLinedOnAxis(filteredVerts) == False:
            ScaleTo0OnAxisAndCursor(filteredVerts, vertsDict, cursorClosestTo)
            return SuccessFinished(me, startTime)

        MakeEqualDistanceBetweenVertsInLine(filteredVerts, vertsDict, cursorClosestTo)
        return SuccessFinished(me, startTime)

    # deselect non quads
    for nf in nonQuadFaces:
        for l in nf.loops:
            set_uv_vert_selected(l, uv_layer, False)

    def isFaceSelected(f):
        return f.select and all(is_uv_vert_selected(l, uv_layer) for l in f.loops)

    def getIslandFromFace(startFace):
        island = set()
        toCheck = set([startFace])

        while (len(toCheck)):
            face = toCheck.pop()
            if isFaceSelected(face) and face not in island:
                island.add(face)
                adjacentFaces = []
                for e in face.edges:
                    if e.seam == False:
                        for f in e.link_faces:
                            if f != face:
                                adjacentFaces.append(f)
                toCheck.update(adjacentFaces)

        return island

    def getIslandsFromSelectedFaces(selectedFaces):
        islands = []
        toCheck = set(selectedFaces)
        while(len(toCheck)):
            face = toCheck.pop()
            island = getIslandFromFace(face)
            islands.append(island)
            toCheck.difference_update(island)
        return islands

    islands = getIslandsFromSelectedFaces(selFaces)

    def main2 (targetFace, faces):
        ShapeFace(uv_layer, operator, targetFace, vertsDict, square)

        if square: FollowActiveUV(operator, me, targetFace, faces, 'EVEN')
        else: FollowActiveUV(operator, me, targetFace, faces)

    for island in islands:
        targetFace = bm.faces.active
        if (targetFace == None or
            targetFace not in island or
            len(islands) > 1 or
            targetFace.select == False or
            len(targetFace.verts) != 4):
                targetFace = next(iter(island))

        main2(targetFace, island)

    if noEdge == False:
        #edge has ripped so we connect it back
        for ev in edgeVerts:
            key = (round(ev.uv.x, precision), round(ev.uv.y, precision))
            if key in vertsDict:
                ev.uv = vertsDict[key][0].uv
                # Note: ev is a BMLoopUV, need to find the loop to set selection
                # This will be handled by the fact that vertsDict syncs positions

    return SuccessFinished(me, startTime)

'''def ScaleSelection(factor, pivot = 'CURSOR'):
    last_pivot = bpy.context.space_data.pivot_point
    bpy.context.space_data.pivot_point = pivot
    bpy.ops.transform.resize(value=(factor, factor, factor), constraint_axis=(False, False, False), mirror=False, proportional_edit_falloff='SMOOTH', proportional_size=1)
    bpy.context.space_data.pivot_point = last_pivot
    return'''

def ShapeFace(uv_layer, operator, targetFace, vertsDict, square):
    corners = []
    for l in targetFace.loops:
        luv = l[uv_layer]
        corners.append(luv)

    if len(corners) != 4:
        #operator.report({'ERROR'}, "bla")
        return

    lucv, ldcv, rucv, rdcv = Corners(corners)

    cct = CursorClosestTo([lucv, ldcv, rdcv, rucv])
    MakeUvFaceEqualRectangle(vertsDict, lucv, rucv, rdcv, ldcv, cct, square)
    return

def MakeUvFaceEqualRectangle(vertsDict, lucv, rucv, rdcv, ldcv, startv, square = False):
    sizeX, sizeY = ImageSize()
    ratio = sizeX/sizeY

    if startv == None: startv = lucv.uv
    elif AreVertsQuasiEqual(startv, rucv): startv = rucv.uv
    elif AreVertsQuasiEqual(startv, rdcv): startv = rdcv.uv
    elif AreVertsQuasiEqual(startv, ldcv): startv = ldcv.uv
    else: startv = lucv.uv

    lucv = lucv.uv
    rucv = rucv.uv
    rdcv = rdcv.uv
    ldcv = ldcv.uv

    if (startv == lucv):
        finalScaleX = hypotVert(lucv, rucv)
        finalScaleY = hypotVert(lucv, ldcv)
        currRowX = lucv.x
        currRowY = lucv.y

    elif (startv == rucv):
        finalScaleX = hypotVert(rucv, lucv)
        finalScaleY = hypotVert(rucv, rdcv)
        currRowX = rucv.x - finalScaleX
        currRowY = rucv.y

    elif (startv == rdcv):
        finalScaleX = hypotVert(rdcv, ldcv)
        finalScaleY = hypotVert(rdcv, rucv)
        currRowX = rdcv.x - finalScaleX
        currRowY = rdcv.y + finalScaleY

    else:
        finalScaleX = hypotVert(ldcv, rdcv)
        finalScaleY = hypotVert(ldcv, lucv)
        currRowX = ldcv.x
        currRowY = ldcv.y +finalScaleY

    if square: finalScaleY = finalScaleX*ratio
    #lucv, rucv
    x = round(lucv.x, precision)
    y = round(lucv.y, precision)
    for v in vertsDict[(x,y)]:
        v.uv.x = currRowX
        v.uv.y = currRowY

    x = round(rucv.x, precision)
    y = round(rucv.y, precision)
    for v in vertsDict[(x,y)]:
        v.uv.x = currRowX + finalScaleX
        v.uv.y = currRowY

    #rdcv, ldcv
    x = round(rdcv.x, precision)
    y = round(rdcv.y, precision)
    for v in vertsDict[(x,y)]:
        v.uv.x = currRowX + finalScaleX
        v.uv.y = currRowY - finalScaleY

    x = round(ldcv.x, precision)
    y = round(ldcv.y, precision)
    for v in vertsDict[(x,y)]:
        v.uv.x = currRowX
        v.uv.y = currRowY - finalScaleY


    return

def SnapCursorToClosestSelected(filteredVerts):
    #TODO: snap to closest selected
    if len(filteredVerts) == 1:
        SetAll2dCursorsTo(filteredVerts[0].uv.x, filteredVerts[0].uv.y)

    return

def ListsOfVerts(uv_layer, bm):
    edgeVerts = []
    allEdgeVerts = []
    filteredVerts = []
    selFaces = []
    nonQuadFaces = []
    vertsDict = defaultdict(list)                #dict

    for f in bm.faces:
        isFaceSel = True
        facesEdgeVerts = []
        if (f.select == False):
            continue

        #collect edge verts if any
        for l in f.loops:
            luv = l[uv_layer]
            if is_uv_vert_selected(l, uv_layer):
                facesEdgeVerts.append(luv)
            else: isFaceSel = False

        allEdgeVerts.extend(facesEdgeVerts)
        if isFaceSel:
            if len(f.verts) != 4:
                nonQuadFaces.append(f)
                edgeVerts.extend(facesEdgeVerts)
            else:
                selFaces.append(f)

                for l in f.loops:
                    luv = l[uv_layer]
                    x = round(luv.uv.x, precision)
                    y = round(luv.uv.y, precision)
                    vertsDict[(x, y)].append(luv)

        else: edgeVerts.extend(facesEdgeVerts)

    noEdge = False
    if len(edgeVerts) == 0:
        noEdge = True
        edgeVerts.extend(allEdgeVerts)

    if len(selFaces) == 0:
        for ev in edgeVerts:
            if ListQuasiContainsVect(filteredVerts, ev) == False:
                filteredVerts.append(ev)
    else: filteredVerts = edgeVerts

    return edgeVerts, filteredVerts, selFaces, nonQuadFaces, vertsDict, noEdge

def ListQuasiContainsVect(list, vect):
    for v in list:
        if AreVertsQuasiEqual(v, vect):
            return True
    return False

#modified ideasman42's uvcalc_follow_active.py
def FollowActiveUV(operator, me, f_act, faces, EXTEND_MODE = 'LENGTH_AVERAGE'):
    bm = bmesh.from_edit_mesh(me)
    uv_act = bm.loops.layers.uv.active

    # our own local walker
    def walk_face_init(faces, f_act):
        # first tag all faces True (so we dont uvmap them)
        for f in bm.faces:
            f.tag = True
        # then tag faces arg False
        for f in faces:
            f.tag = False
        # tag the active face True since we begin there
        f_act.tag = True

    def walk_face(f):
        # all faces in this list must be tagged
        f.tag = True
        faces_a = [f]
        faces_b = []

        while faces_a:
            for f in faces_a:
                for l in f.loops:
                    l_edge = l.edge
                    if (l_edge.is_manifold == True) and (l_edge.seam == False):
                        l_other = l.link_loop_radial_next
                        f_other = l_other.face
                        if not f_other.tag:
                            yield (f, l, f_other)
                            f_other.tag = True
                            faces_b.append(f_other)
            # swap
            faces_a, faces_b = faces_b, faces_a
            faces_b.clear()

    def walk_edgeloop(l):
        """
        Could make this a generic function
        """
        e_first = l.edge
        e = None
        while True:
            e = l.edge
            yield e

            # don't step past non-manifold edges
            if e.is_manifold:
                # welk around the quad and then onto the next face
                l = l.link_loop_radial_next
                if len(l.face.verts) == 4:
                    l = l.link_loop_next.link_loop_next
                    if l.edge == e_first:
                        break
                else:
                    break
            else:
                break

    def extrapolate_uv(fac,
                       l_a_outer, l_a_inner,
                       l_b_outer, l_b_inner):
        l_b_inner[:] = l_a_inner
        l_b_outer[:] = l_a_inner + ((l_a_inner - l_a_outer) * fac)

    def apply_uv(f_prev, l_prev, f_next):
        l_a = [None, None, None, None]
        l_b = [None, None, None, None]

        l_a[0] = l_prev
        l_a[1] = l_a[0].link_loop_next
        l_a[2] = l_a[1].link_loop_next
        l_a[3] = l_a[2].link_loop_next

        #  l_b
        #  +-----------+
        #  |(3)        |(2)
        #  |           |
        #  |l_next(0)  |(1)
        #  +-----------+
        #        ^
        #  l_a   |
        #  +-----------+
        #  |l_prev(0)  |(1)
        #  |    (f)    |
        #  |(3)        |(2)
        #  +-----------+
        #  copy from this face to the one above.

        # get the other loops
        l_next = l_prev.link_loop_radial_next
        if l_next.vert != l_prev.vert:
            l_b[1] = l_next
            l_b[0] = l_b[1].link_loop_next
            l_b[3] = l_b[0].link_loop_next
            l_b[2] = l_b[3].link_loop_next
        else:
            l_b[0] = l_next
            l_b[1] = l_b[0].link_loop_next
            l_b[2] = l_b[1].link_loop_next
            l_b[3] = l_b[2].link_loop_next

        l_a_uv = [l[uv_act].uv for l in l_a]
        l_b_uv = [l[uv_act].uv for l in l_b]

        if EXTEND_MODE == 'LENGTH_AVERAGE':
            try:
                fac = edge_lengths[l_b[2].edge.index][0] / edge_lengths[l_a[1].edge.index][0]
            except ZeroDivisionError:
                fac = 1.0
        elif EXTEND_MODE == 'LENGTH':
            a0, b0, c0 = l_a[3].vert.co, l_a[0].vert.co, l_b[3].vert.co
            a1, b1, c1 = l_a[2].vert.co, l_a[1].vert.co, l_b[2].vert.co

            d1 = (a0 - b0).length + (a1 - b1).length
            d2 = (b0 - c0).length + (b1 - c1).length
            try:
                fac = d2 / d1
            except ZeroDivisionError:
                fac = 1.0
        else:
            fac = 1.0

        extrapolate_uv(fac,
                       l_a_uv[3], l_a_uv[0],
                       l_b_uv[3], l_b_uv[0])

        extrapolate_uv(fac,
                       l_a_uv[2], l_a_uv[1],
                       l_b_uv[2], l_b_uv[1])

    # -------------------------------------------
    # Calculate average length per loop if needed

    if EXTEND_MODE == 'LENGTH_AVERAGE':
        bm.edges.index_update()
        edge_lengths = [None] * len(bm.edges)   #NoneType times the length of edges list

        for f in faces:
            # we know its a quad
            l_quad = f.loops[:]
            l_pair_a = (l_quad[0], l_quad[2])
            l_pair_b = (l_quad[1], l_quad[3])

            for l_pair in (l_pair_a, l_pair_b):
                if edge_lengths[l_pair[0].edge.index] == None:

                    edge_length_store = [-1.0]
                    edge_length_accum = 0.0
                    edge_length_total = 0

                    for l in l_pair:
                        if edge_lengths[l.edge.index] == None:
                            for e in walk_edgeloop(l):
                                if edge_lengths[e.index] == None:
                                    edge_lengths[e.index] = edge_length_store
                                    edge_length_accum += e.calc_length()
                                    edge_length_total += 1

                    edge_length_store[0] = edge_length_accum / edge_length_total

    # done with average length
    # ------------------------

    walk_face_init(faces, f_act)
    for f_triple in walk_face(f_act):
        apply_uv(*f_triple)

    bmesh.update_edit_mesh(me, loop_triangles=False)

'''----------------------------------'''

def SuccessFinished(me, startTime):
    #use for backtrack of steps
    #bpy.ops.ed.undo_push()
    bmesh.update_edit_mesh(me)
    elapsed = round(timer()-startTime, 2)
    #if (elapsed >= 0.05): operator.report({'INFO'}, "UvSquares finished, elapsed:", elapsed, "s.")
    if (elapsed >= 0.05): print("UvSquares finished, elapsed:", elapsed, "s.")
    return

'''def SymmetrySelected(axis, pivot = "MEDIAN"):
    last_pivot = bpy.context.space_data.pivot_point
    bpy.context.space_data.pivot_point = pivot
    bpy.ops.transform.mirror(constraint_axis=(True, False, False), constraint_orientation='GLOBAL', proportional_edit_falloff='SMOOTH', proportional_size=1)
    bpy.context.space_data.pivot_point = last_pivot
    return'''

def AreVectsLinedOnAxis(verts):
    areLinedX = True
    areLinedY = True
    allowedError = 0.00001
    valX = verts[0].uv.x
    valY = verts[0].uv.y
    for v in verts:
        if abs(valX - v.uv.x) > allowedError:
            areLinedX = False
        if abs(valY - v.uv.y) > allowedError:
            areLinedY = False
    return areLinedX or areLinedY

def MakeEqualDistanceBetweenVertsInLine(filteredVerts, vertsDict, startv = None):
    verts = filteredVerts
    verts.sort(key=lambda x: x.uv[0])      #sort by .x

    first = verts[0].uv
    last = verts[len(verts)-1].uv

    horizontal = True
    if ((last.x - first.x) >0.00001):
        slope = (last.y - first.y)/(last.x - first.x)
        if (slope > 1) or (slope <-1):
            horizontal = False
    else:
        horizontal = False

    if horizontal == True:
        length = hypot(first.x - last.x, first.y - last.y)

        if startv == last:
            currentX = last.x - length
            currentY = last.y
        else:
            currentX = first.x
            currentY = first.y
    else:
        verts.sort(key=lambda x: x.uv[1])  #sort by .y
        verts.reverse()     #reverse because y values drop from up to down
        first = verts[0].uv
        last = verts[len(verts)-1].uv

        length = hypot(first.x - last.x, first.y - last.y)  # we have to call length here because if it is not Hor first and second can not actually be first and second

        if startv == last:
            currentX = last.x
            currentY = last.y + length

        else:
            currentX = first.x
            currentY = first.y

    numberOfVerts = len(verts)
    finalScale = length / (numberOfVerts-1)

    if horizontal == True:
        first = verts[0]
        last = verts[len(verts)-1]

        for v in verts:
            v = v.uv
            x = round(v.x, precision)
            y = round(v.y, precision)

            for vert in vertsDict[(x,y)]:
                vert.uv.x = currentX
                vert.uv.y = currentY

            currentX = currentX + finalScale
    else:
        for v in verts:
            x = round(v.uv.x, precision)
            y = round(v.uv.y, precision)

            for vert in vertsDict[(x,y)]:
                vert.uv.x = currentX
                vert.uv.y = currentY

            currentY = currentY - finalScale
    return

def VertsDictForLine(uv_layer, bm, selVerts, vertsDict):
    for f in bm.faces:
        for l in f.loops:
                luv = l[uv_layer]
                if is_uv_vert_selected(l, uv_layer):
                    x = round(luv.uv.x, precision)
                    y = round(luv.uv.y, precision)

                    vertsDict[(x, y)].append(luv)
    return

def ScaleTo0OnAxisAndCursor(filteredVerts, vertsDict, startv = None, horizontal = None):

    verts = filteredVerts
    verts.sort(key=lambda x: x.uv[0])      #sort by .x

    first = verts[0]
    last = verts[len(verts)-1]

    if horizontal == None:
        horizontal = True
        if ((last.uv.x - first.uv.x) >0.00001):
            slope = (last.uv.y - first.uv.y)/(last.uv.x - first.uv.x)
            if (slope > 1) or (slope <-1):
                horizontal = False
        else:
            horizontal = False

    if horizontal == True:
        if startv == None:
            startv = first

        SetAll2dCursorsTo(startv.uv.x, startv.uv.y)
        #scale to 0 on Y
        ScaleTo0('Y')
        return

    else:
        verts.sort(key=lambda x: x.uv[1])  #sort by .y
        verts.reverse()     #reverse because y values drop from up to down
        first = verts[0]
        last = verts[len(verts)-1]
        if startv == None:
            startv = first

        SetAll2dCursorsTo(startv.uv.x, startv.uv.y)
        #scale to 0 on X
        ScaleTo0('X')
        return

def ScaleTo0(axis):
    last_pivot = bpy.context.space_data.pivot_point
    bpy.context.space_data.pivot_point = 'CURSOR'

    for area in bpy.context.screen.areas:
        if area.type == 'IMAGE_EDITOR':
            if axis == 'Y':
                bpy.ops.transform.resize(value=(1, 0, 1), constraint_axis=(False, True, False), mirror=False, proportional_edit_falloff='SMOOTH', proportional_size=1)
            else:
                bpy.ops.transform.resize(value=(0, 1, 1), constraint_axis=(True, False, False), mirror=False, proportional_edit_falloff='SMOOTH', proportional_size=1)


    bpy.context.space_data.pivot_point = last_pivot
    return


def hypotVert(v1, v2):
    hyp = hypot(v1.x - v2.x, v1.y - v2.y)
    return hyp

def Corners(corners):
    firstHighest = corners[0]
    for c in corners:
        if c.uv.y > firstHighest.uv.y:
            firstHighest = c
    corners.remove(firstHighest)

    secondHighest = corners[0]
    for c in corners:
        if (c.uv.y > secondHighest.uv.y):
            secondHighest = c

    if firstHighest.uv.x < secondHighest.uv.x:
        leftUp = firstHighest
        rightUp = secondHighest
    else:
        leftUp = secondHighest
        rightUp = firstHighest
    corners.remove(secondHighest)

    firstLowest = corners[0]
    secondLowest = corners[1]

    if firstLowest.uv.x < secondLowest.uv.x:
        leftDown = firstLowest
        rightDown = secondLowest
    else:
        leftDown = secondLowest
        rightDown = firstLowest

    return leftUp, leftDown, rightUp, rightDown

def ImageSize():
    ratioX, ratioY = 256,256
    for a in bpy.context.screen.areas:
        if a.type == 'IMAGE_EDITOR':
            img = a.spaces[0].image
            if img != None and img.size[0] != 0:
                ratioX, ratioY = img.size[0], img.size[1]
            break
    return ratioX, ratioY

def CursorClosestTo(verts):
    sizeX, sizeY = ImageSize()
    if bpy.app.version >= (2, 80, 0):
        sizeX, sizeY = 1,1
    min = float('inf')
    minV = verts[0]
    for v in verts:
        if v == None: continue
        for area in bpy.context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                loc = area.spaces[0].cursor_location
                hyp = hypot(loc.x/sizeX -v.uv.x, loc.y/sizeY -v.uv.y)
                if (hyp < min):
                    min = hyp
                    minV = v
    return minV

def SetAll2dCursorsTo(x,y):
    bpy.ops.uv.cursor_set(location=(x, y))
    return

def AreVertsQuasiEqual(v1, v2, allowedError = 0.00001):
    if abs(v1.uv.x -v2.uv.x) < allowedError and abs(v1.uv.y -v2.uv.y) < allowedError:
        return True
    return False



class NASH3D_OT_uv_squares(bpy.types.Operator):
    """Reshapes UV faces to a grid of equivalent squares"""
    bl_idname = "nash3d.uv_squares"
    bl_label = "UVs to Grid of Squares"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.active_object and 
                context.active_object.type == 'MESH' and 
                context.active_object.mode == 'EDIT')

    def execute(self, context):
        # We need to temporarily disable sync mode if it's on
        sync_mode = context.scene.tool_settings.use_uv_select_sync
        if sync_mode:
            context.scene.tool_settings.use_uv_select_sync = False
            
        try:
            main(context, self, True)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to run UV Squares: {str(e)}")
            return {'CANCELLED'}
        finally:
            if sync_mode:
                context.scene.tool_settings.use_uv_select_sync = True
                
        return {'FINISHED'}

# =========================================================================
# SCALE ISLANDS OPERATORS
# =========================================================================

def _get_uv_island_loops_for_face(bm, uv_layer, face):
    """Return loops of the active-face island and lists of loops for all other selected islands."""
    islands = get_uv_islands(bm, uv_layer)
    
    active_island = None
    if face and face.loops:
        first_loop = face.loops[0]
        for island in islands:
            if first_loop in island:
                active_island = island
                break
                
    other_islands = []
    if active_island:
        for island in islands:
            if island is active_island:
                continue
            # Check if island is selected (any loop selected)
            is_selected = False
            for l in island:
                if (l.uv_select_vert if hasattr(l, "uv_select_vert") else l[uv_layer].select):
                    is_selected = True
                    break
            if is_selected:
                other_islands.append(island)
                
    return active_island, other_islands


def _island_bbox(loops, uv_layer):
    xs = [l[uv_layer].uv.x for l in loops]
    ys = [l[uv_layer].uv.y for l in loops]
    return min(xs), min(ys), max(xs), max(ys)


class NASH3D_OT_orient_uvs_by_world(bpy.types.Operator):
    """Orient the selected UV islands based on the orientation of the world"""
    bl_idname = "nash3d.orient_uvs_by_world"
    bl_label = "Orient UVs by World"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.active_object and
                context.active_object.type == 'MESH' and
                context.active_object.mode == 'EDIT')

    def execute(self, context):
        import bmesh
        import math
        import mathutils

        obj = context.active_object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            self.report({'ERROR'}, "No active UV layer found.")
            return {'CANCELLED'}

        matrix_world = obj.matrix_world
        normal_matrix = matrix_world.inverted_safe().transposed().to_3x3()

        islands = get_uv_islands(bm, uv_layer)
        
        processed_count = 0
        for island in islands:
            sel_loops = [l for l in island if (l.uv_select_vert if hasattr(l, "uv_select_vert") else l[uv_layer].select)]
            if not sel_loops:
                continue

            sum_A = 0.0
            sum_B = 0.0

            # Precalculate island centroid to rotate around
            u_sum = sum(l[uv_layer].uv.x for l in sel_loops)
            v_sum = sum(l[uv_layer].uv.y for l in sel_loops)
            cx = u_sum / len(sel_loops)
            cy = v_sum / len(sel_loops)

            # Accumulate rotation vectors for all edges in this island
            for l in island:
                face = l.face
                
                # World space normal
                N = normal_matrix @ face.normal
                if N.length_squared < 1e-8:
                    continue
                N.normalize()
                
                Z_world = mathutils.Vector((0, 0, 1))
                Y_world = mathutils.Vector((0, 1, 0))
                
                # Calculate Up vector
                # Use a 45 degree threshold to distinguish walls from roofs/floors
                if abs(N.z) < 0.7071:
                    Up = Z_world - Z_world.dot(N) * N
                elif N.z > 0:
                    Up = Y_world - Y_world.dot(N) * N
                else:
                    Up = -Y_world - (-Y_world).dot(N) * N
                    
                if Up.length_squared < 1e-8:
                    continue
                Up.normalize()
                
                Right = Up.cross(N)
                Right.normalize()
                
                # 3D edge vector in world space
                p1 = matrix_world @ l.vert.co
                p2 = matrix_world @ l.link_loop_next.vert.co
                E3 = p2 - p1
                
                if E3.length_squared < 1e-12:
                    continue
                
                # Ideal UV direction for this edge
                du_ideal = E3.dot(Right)
                dv_ideal = E3.dot(Up)
                
                # Actual UV direction for this edge
                uv1 = l[uv_layer].uv
                uv2 = l.link_loop_next[uv_layer].uv
                du_uv = uv2.x - uv1.x
                dv_uv = uv2.y - uv1.y
                
                # Accumulate
                sum_A += du_uv * du_ideal + dv_uv * dv_ideal
                sum_B += du_uv * dv_ideal - dv_uv * du_ideal

            if abs(sum_A) < 1e-8 and abs(sum_B) < 1e-8:
                continue
                
            # Optimal rotation angle
            theta = math.atan2(sum_B, sum_A)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            # Rotate all UVs in the island
            for l in island:
                u = l[uv_layer].uv.x - cx
                v = l[uv_layer].uv.y - cy
                
                l[uv_layer].uv.x = cx + u * cos_t - v * sin_t
                l[uv_layer].uv.y = cy + u * sin_t + v * cos_t

            processed_count += 1

        if processed_count > 0:
            bmesh.update_edit_mesh(me)
            self.report({'INFO'}, f"Oriented {processed_count} island(s) by world.")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "No selected UV islands found.")
            return {'CANCELLED'}


class NASH3D_OT_scale_islands_by_x(bpy.types.Operator):
    """Uniformly scale all selected UV islands to match the X width of the active-face island"""
    bl_idname = "nash3d.scale_islands_by_x"
    bl_label = "Scale Islands by X"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.active_object and
                context.active_object.type == 'MESH' and
                context.active_object.mode == 'EDIT')

    def execute(self, context):
        obj = context.active_object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.verify()

        active_face = bm.faces.active
        if active_face is None:
            self.report({'WARNING'}, "No active face selected.")
            return {'CANCELLED'}

        active_island, other_islands = _get_uv_island_loops_for_face(bm, uv_layer, active_face)
        
        if not active_island:
            self.report({'WARNING'}, "Could not find active island.")
            return {'CANCELLED'}

        bb_ref = _island_bbox(active_island, uv_layer)
        ref_width = bb_ref[2] - bb_ref[0]

        if ref_width < 1e-7:
            self.report({'WARNING'}, "Active island has zero width.")
            return {'CANCELLED'}

        for island in other_islands:
            bb = _island_bbox(island, uv_layer)
            isl_width = bb[2] - bb[0]
            if isl_width < 1e-7:
                continue
            # Uniform scale factor derived from X axis ratio
            scale = ref_width / isl_width
            pivot_x, pivot_y = bb[0], bb[1]
            for l in island:
                u = l[uv_layer].uv.x
                v = l[uv_layer].uv.y
                l[uv_layer].uv.x = pivot_x + (u - pivot_x) * scale
                l[uv_layer].uv.y = pivot_y + (v - pivot_y) * scale

        bmesh.update_edit_mesh(me)
        self.report({'INFO'}, f"Scaled {len(other_islands)} island(s) to match X width ({ref_width:.4f}).")
        return {'FINISHED'}


class NASH3D_OT_scale_islands_by_y(bpy.types.Operator):
    """Uniformly scale all selected UV islands to match the Y height of the active-face island"""
    bl_idname = "nash3d.scale_islands_by_y"
    bl_label = "Scale Islands by Y"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.active_object and
                context.active_object.type == 'MESH' and
                context.active_object.mode == 'EDIT')

    def execute(self, context):
        obj = context.active_object
        me = obj.data
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.verify()

        active_face = bm.faces.active
        if active_face is None:
            self.report({'WARNING'}, "No active face selected.")
            return {'CANCELLED'}

        active_island, other_islands = _get_uv_island_loops_for_face(bm, uv_layer, active_face)
        
        if not active_island:
            self.report({'WARNING'}, "Could not find active island.")
            return {'CANCELLED'}

        bb_ref = _island_bbox(active_island, uv_layer)
        ref_height = bb_ref[3] - bb_ref[1]

        if ref_height < 1e-7:
            self.report({'WARNING'}, "Active island has zero height.")
            return {'CANCELLED'}

        for island in other_islands:
            bb = _island_bbox(island, uv_layer)
            isl_height = bb[3] - bb[1]
            if isl_height < 1e-7:
                continue
            # Uniform scale factor derived from Y axis ratio
            scale = ref_height / isl_height
            pivot_x, pivot_y = bb[0], bb[1]
            for l in island:
                u = l[uv_layer].uv.x
                v = l[uv_layer].uv.y
                l[uv_layer].uv.x = pivot_x + (u - pivot_x) * scale
                l[uv_layer].uv.y = pivot_y + (v - pivot_y) * scale

        bmesh.update_edit_mesh(me)
        self.report({'INFO'}, f"Scaled {len(other_islands)} island(s) to match Y height ({ref_height:.4f}).")
        return {'FINISHED'}


class IMAGE_PT_oshan_uv_deforming(bpy.types.Panel):

    bl_label = "Deforming Tools"
    bl_idname = "IMAGE_PT_oshan_uv_deforming"
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
        
        if not (obj and obj.type == 'MESH' and obj.mode == 'EDIT'):
            return
            
        col = layout.column(align=True)
        col.operator("nash3d.uv_squares", text="UV Squares", icon='GRID')
        col.operator("nash3d.orient_uvs_by_world", text="Orient UVs by World", icon='ORIENTATION_GLOBAL')
        col.separator(factor=0.5)
        row = col.row(align=True)
        row.operator("nash3d.scale_islands_by_x", text="Scale by X", icon='DRIVER_DISTANCE')
        row.operator("nash3d.scale_islands_by_y", text="Scale by Y", icon='DRIVER_DISTANCE')

# =========================================================================
# TEXTURES SUBPANEL & OPERATORS
# =========================================================================
class NASH3D_RecentImageItem(bpy.types.PropertyGroup):
    """Holds a single reference to a recently used image."""
    image: bpy.props.PointerProperty(
        name="Image",
        type=bpy.types.Image
    )

def push_recent_image(scene, image):
    """Push an image into the scene's recent images ring buffer (max 5)."""
    if not image:
        return
        
    history = scene.oshan_recent_images
    max_slots = 5

    dup_index = -1
    for i, item in enumerate(history):
        if item.image == image:
            dup_index = i
            break

    if dup_index >= 0:
        history.move(dup_index, 0)
    else:
        history.add()
        history.move(len(history) - 1, 0)
        history[0].image = image

        while len(history) > max_slots:
            history.remove(len(history) - 1)

def on_uv_image_picker_update(self, context):
    img = self.oshan_uv_image_picker
    if img:
        push_recent_image(self, img)
        if context.space_data and context.space_data.type == 'IMAGE_EDITOR':
            context.space_data.image = img

class NASH3D_OT_set_uv_background_image(bpy.types.Operator):
    """Set the clicked image as the UV Editor background"""
    bl_idname = "nash3d.set_uv_background_image"
    bl_label = "Set UV Background"
    bl_options = {'INTERNAL'}

    index: bpy.props.IntProperty(default=0)

    def execute(self, context):
        scene = context.scene
        history = scene.oshan_recent_images
        
        if 0 <= self.index < len(history):
            img = history[self.index].image
            if img:
                scene.oshan_uv_image_picker = img
                
        return {'FINISHED'}

class IMAGE_PT_oshan_uv_textures(bpy.types.Panel):
    bl_label = "Textures"
    bl_idname = "IMAGE_PT_oshan_uv_textures"
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
        scene = context.scene
        
        col = layout.column(align=True)
        col.prop(scene, "oshan_uv_image_picker", text="")
        
        history = scene.oshan_recent_images
        if len(history) > 0:
            col.separator(factor=1.0)
            col.label(text="Recently Used:")
            for i, item in enumerate(history):
                if item.image:
                    row = col.row(align=True)
                    op = row.operator("nash3d.set_uv_background_image", text=item.image.name, icon_value=layout.icon(item.image))
                    op.index = i

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
    NASH3D_OT_remove_all_materials,
    NASH3D_OT_copy_materials,
    VIEW3D_PT_oshan_vertex_paint_subpanel,
    VIEW3D_PT_oshan_misc_tools_subpanel,
    VIEW3D_PT_oshan_about_subpanel,
    IMAGE_PT_oshan_uv_tools,
    IMAGE_PT_oshan_uv_snapping,
    IMAGE_PT_oshan_uv_rotating,
    IMAGE_PT_oshan_texel_density,
    NASH3D_OT_get_texel_density,
    NASH3D_OT_explode_islands,
    NASH3D_OT_uv_squares,
    NASH3D_OT_orient_uvs_by_world,
    NASH3D_OT_scale_islands_by_x,
    NASH3D_OT_scale_islands_by_y,
    IMAGE_PT_oshan_uv_deforming,
    NASH3D_RecentImageItem,
    NASH3D_OT_set_uv_background_image,
    IMAGE_PT_oshan_uv_textures,
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

    # Register recent UV images
    bpy.types.Scene.oshan_recent_images = bpy.props.CollectionProperty(
        type=NASH3D_RecentImageItem,
        name="Recent UV Images",
        description="Last 5 images used as UV background"
    )
    bpy.types.Scene.oshan_uv_image_picker = bpy.props.PointerProperty(
        type=bpy.types.Image,
        name="Select Image",
        description="Select an image to set as background and add to recent",
        update=on_uv_image_picker_update
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

    # Delete recent UV images properties
    del bpy.types.Scene.oshan_recent_images
    del bpy.types.Scene.oshan_uv_image_picker

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
