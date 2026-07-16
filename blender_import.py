"""
Blender add-on: import perspective-sculptor pieces.json as Bezier curves.

Installation
------------
    1. Blender: Edit > Preferences > Add-ons > Install...
    2. Select this file (blender_import.py) and enable
       "Import-Export: Perspective Sculptor Importer".
    3. In the 3D Viewport press N to open the sidebar and switch to the
       "Sculptor" tab.
    4. Set the path to your pieces.json and click "Import Pieces".

The file can also still be run directly from the Scripting workspace
("Run Script" registers the add-on for the current session).

Shape fidelity
--------------
Each piece is imported as a native Bezier curve. The handleIn/handleOut
vectors stored in the JSON become the spline's handle_left/handle_right
(kept as FREE handles), so the imported outline is the exact same cubic
Bezier the app exported — no sampling or polyline approximation.

Coordinate mapping
------------------
The Python app uses Y-up (OpenGL).  Blender uses Z-up.
This add-on maps  app(x, y, z)  →  Blender(x, z, y)  via each object's
matrix_world, so the curve control points stay in local 2-D space exactly
as they are stored in the JSON.

The piece's Y-axis rotation (theta) from app space becomes:

    Blender matrix_world =
        [[cos θ,  0, -sin θ,  cx_app],      columns = local-X, local-Y,
         [-sin θ, 0, -cos θ,  cz_app],               local-Z, translation
         [0,      1,  0,      cy_app],
         [0,      0,  0,      1     ]]

For a local point (lx, ly, 0) this gives Blender world position:
    (cos θ · lx + cx,  -sin θ · lx + cz,  ly + cy)
which equals  app_to_blender( R_y(θ) @ (lx, ly, 0) + centre ).
"""

import bpy
import json
import math
from mathutils import Matrix, Vector

bl_info = {
    "name": "Perspective Sculptor Importer",
    "author": "automated-perspective-sculptor",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Sculptor",
    "description": "Import perspective-sculptor pieces.json as Bezier curves",
    "category": "Import-Export",
}

# Matches Patch.DEFAULT_THICKNESS in core/patch.py
DEFAULT_THICKNESS = 0.035


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_collection(name: str):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def clear_collection(col):
    """Remove all objects in `col`, along with their curve data."""
    for obj in list(col.objects):
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            bpy.data.curves.remove(data)


def hex_to_rgb(h: str):
    h = h.lstrip('#')
    return (int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0)


def get_material(r: float, g: float, b: float, name: str):
    """Fetch or create a Principled material (reused across re-imports)."""
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.6
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    return mat


def piece_matrix(theta: float, cx: float, cy: float, cz: float) -> Matrix:
    """
    Build the 4×4 world matrix for one piece.

    Applies the Y-axis rotation from app space (matching core/patch.py rot_y)
    and remaps the coordinate system from app Y-up to Blender Z-up in one step.

    Derivation:
        R_y(θ) in app: [[c,0,s],[0,1,0],[-s,0,c]]
        app_to_blender: (x,y,z) → (x,z,y)
        Combined: local (lx,ly,0) → Blender (c·lx, -s·lx, ly) + (cx, cz, cy)
    """
    c, s = math.cos(theta), math.sin(theta)
    rot = Matrix([
        [ c,  0, -s],
        [-s,  0, -c],
        [ 0,  1,  0],
    ])
    mat = rot.to_4x4()
    mat.translation = Vector((cx, cz, cy))   # app(x,y,z) → Blender(x,z,y)
    return mat


def build_spline(crv, point_triples, closed: bool = True):
    """
    Populate a new BEZIER spline on `crv` from a list of
    (co_xy, handle_left_xy, handle_right_xy) tuples.
    All coordinates are 2-D (x, y); Z is set to 0.
    Handles are kept FREE so the exported handleIn/handleOut vectors are
    preserved verbatim and the curve shape matches the app exactly.
    """
    pts = list(point_triples)
    sp = crv.splines.new('BEZIER')
    sp.bezier_points.add(len(pts) - 1)   # already has 1 point
    sp.use_cyclic_u = closed

    for i, (co, hl, hr) in enumerate(pts):
        bp = sp.bezier_points[i]
        bp.co           = Vector((co[0], co[1], 0.0))
        bp.handle_left  = Vector((hl[0], hl[1], 0.0))
        bp.handle_right = Vector((hr[0], hr[1], 0.0))
        bp.handle_left_type  = 'FREE'
        bp.handle_right_type = 'FREE'


def piece_spline_points(cps: list):
    """
    Yield (co, handle_left, handle_right) in local 2-D piece space.

    JSON stores control-point positions with anchor_xy already subtracted.
    handleIn  → incoming handle → Blender handle_left
    handleOut → outgoing handle → Blender handle_right
    Handles are stored as VECTORS; add them to the anchor position.
    """
    for cp in cps:
        lx  = float(cp['x']);  ly  = float(cp['y'])
        hix = float(cp['handleIn']['x']);  hiy = float(cp['handleIn']['y'])
        hox = float(cp['handleOut']['x']); hoy = float(cp['handleOut']['y'])
        yield (
            (lx,        ly),           # anchor
            (lx + hix,  ly + hiy),     # handle_left  (= anchor + handleIn)
            (lx + hox,  ly + hoy),     # handle_right (= anchor + handleOut)
        )


def circle_points(cx: float, cy: float, r: float):
    """
    Yield 4 (co, handle_left, handle_right) tuples approximating a circle
    of radius `r` centred at (cx, cy) using cubic Bezier segments.
    Magic constant k ≈ 0.5523 gives the best circular approximation.
    """
    k = r * 0.5523
    yield ((cx + r, cy),     (cx + r, cy - k), (cx + r, cy + k))
    yield ((cx,     cy + r), (cx + k, cy + r), (cx - k, cy + r))
    yield ((cx - r, cy),     (cx - r, cy + k), (cx - r, cy - k))
    yield ((cx,     cy - r), (cx - k, cy - r), (cx + k, cy - r))


def add_curve_object(name: str, points, col, mat, matrix, extrude: float):
    """Create a filled 2-D Bezier curve object and link it to `col`."""
    crv = bpy.data.curves.new(name=name, type='CURVE')
    crv.dimensions = '2D'
    crv.fill_mode  = 'BOTH'
    crv.extrude    = extrude          # ± this distance along local Z

    build_spline(crv, points)

    obj = bpy.data.objects.new(name, crv)
    obj.matrix_world = matrix
    col.objects.link(obj)
    if mat is not None:
        crv.materials.append(mat)
    return obj


# ── Import logic ───────────────────────────────────────────────────────────────

def import_pieces_json(
    json_path: str,
    collection_name: str = "Pieces",
    thickness: float = DEFAULT_THICKNESS,
    show_holes: bool = True,
    hole_radius: float = 0.018,
    clear_existing: bool = False,
) -> tuple[int, int]:
    """Import a pieces.json file. Returns (imported_count, skipped_count)."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    pieces = data.get('pieces', [])
    col = get_collection(collection_name)
    if clear_existing:
        clear_collection(col)

    imported = 0
    skipped = 0
    for pdict in pieces:
        pid   = pdict.get('id', 'piece')
        theta = float(pdict.get('theta', 0.0))
        pos   = pdict['position']
        cx    = float(pos['x'])
        cy    = float(pos['y'])
        cz    = float(pos['z'])
        color = pdict.get('color', '#888888')
        cps   = pdict.get('controlPoints', [])

        if len(cps) < 2:
            print(f"  Skipped {pid}: fewer than 2 control points.")
            skipped += 1
            continue

        r, g, b = hex_to_rgb(color)
        mat = get_material(r, g, b, f"Mat_{color}")
        matrix = piece_matrix(theta, cx, cy, cz)

        # Piece outline curve
        add_curve_object(
            name    = pid,
            points  = piece_spline_points(cps),
            col     = col,
            mat     = mat,
            matrix  = matrix,
            extrude = thickness * 0.5,   # Blender extrude is half-thickness
        )
        imported += 1

        # Optional string-hole markers
        if show_holes:
            black = get_material(0.0, 0.0, 0.0, "PS_Hole_Black")
            for conn in pdict.get('stringConnections', []):
                lp  = conn.get('pieceLocalPoint', {})
                hlx = float(lp.get('x', 0.0))
                hly = float(lp.get('y', 0.0))
                add_curve_object(
                    name    = f"{pid}_{conn.get('id', 'hole')}",
                    points  = circle_points(hlx, hly, hole_radius),
                    col     = col,
                    mat     = black,
                    matrix  = matrix,
                    extrude = thickness * 0.5,
                )

    return imported, skipped


# ── Add-on UI ──────────────────────────────────────────────────────────────────

class PSCULPT_Settings(bpy.types.PropertyGroup):
    json_path: bpy.props.StringProperty(
        name="Pieces JSON",
        description="Path to the pieces.json exported by the perspective sculptor",
        subtype='FILE_PATH',
        default="",
    )
    collection_name: bpy.props.StringProperty(
        name="Collection",
        description="Collection the imported pieces are linked into",
        default="Pieces",
    )
    thickness: bpy.props.FloatProperty(
        name="Thickness",
        description="Full piece thickness (matches Patch.DEFAULT_THICKNESS)",
        default=DEFAULT_THICKNESS,
        min=0.0,
        precision=4,
    )
    show_holes: bpy.props.BoolProperty(
        name="String Holes",
        description="Draw circles at string-attachment points",
        default=True,
    )
    hole_radius: bpy.props.FloatProperty(
        name="Hole Radius",
        description="Radius of string-hole marker circles (scene units)",
        default=0.018,
        min=0.0,
        precision=4,
    )
    clear_existing: bpy.props.BoolProperty(
        name="Clear Collection First",
        description="Remove existing objects in the target collection before importing",
        default=True,
    )


class PSCULPT_OT_import_pieces(bpy.types.Operator):
    bl_idname = "psculpt.import_pieces"
    bl_label = "Import Pieces"
    bl_description = "Import the pieces.json file as Bezier curves"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        settings = context.scene.psculpt_settings
        path = bpy.path.abspath(settings.json_path)

        if not path:
            self.report({'ERROR'}, "Set the path to a pieces.json file first.")
            return {'CANCELLED'}

        try:
            imported, skipped = import_pieces_json(
                json_path       = path,
                collection_name = settings.collection_name,
                thickness       = settings.thickness,
                show_holes      = settings.show_holes,
                hole_radius     = settings.hole_radius,
                clear_existing  = settings.clear_existing,
            )
        except FileNotFoundError:
            self.report({'ERROR'}, f"File not found: {path}")
            return {'CANCELLED'}
        except json.JSONDecodeError as exc:
            self.report({'ERROR'}, f"Invalid JSON: {exc}")
            return {'CANCELLED'}
        except (KeyError, TypeError, ValueError) as exc:
            self.report({'ERROR'}, f"Unexpected pieces.json structure: {exc}")
            return {'CANCELLED'}

        msg = f"Imported {imported} piece(s) into '{settings.collection_name}'"
        if skipped:
            msg += f" ({skipped} skipped — see console)"
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class PSCULPT_PT_sidebar(bpy.types.Panel):
    bl_label = "Perspective Sculptor"
    bl_idname = "PSCULPT_PT_sidebar"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Sculptor"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.psculpt_settings

        layout.prop(settings, "json_path")
        layout.prop(settings, "collection_name")
        layout.prop(settings, "thickness")

        row = layout.row()
        row.prop(settings, "show_holes")
        sub = row.row()
        sub.enabled = settings.show_holes
        sub.prop(settings, "hole_radius", text="Radius")

        layout.prop(settings, "clear_existing")
        layout.separator()
        layout.operator(PSCULPT_OT_import_pieces.bl_idname, icon='IMPORT')


classes = (
    PSCULPT_Settings,
    PSCULPT_OT_import_pieces,
    PSCULPT_PT_sidebar,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.psculpt_settings = bpy.props.PointerProperty(type=PSCULPT_Settings)


def unregister():
    del bpy.types.Scene.psculpt_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
