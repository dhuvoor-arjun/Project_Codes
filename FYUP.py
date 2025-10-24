# Install required libraries (run once per environment)
#!pip install ezdxf pandas

import ezdxf
import pandas as pd
from math import atan2,degrees
from shapely.geometry import Polygon, LineString, Point
import json
import pandas as pd
import os
from datetime import datetime

# --- Step 1: Load DXF file ---
# Replace with your actual DXF file name
file_path = "wallthicknessbound.dxf"
doc = ezdxf.readfile(r"C:\Users\dhuvo\Documents\university\FYUP\wallthicknessbound.dxf")
msp = doc.modelspace() 

units = doc.header.get('$INSUNITS', 0)

unit_map = {
    0: "Unitless",
    1: "Inches",
    2: "Feet",
    3: "Miles",
    4: "Millimeters",
    5: "Centimeters",
    6: "Meters",
    7: "Kilometers",
    8: "Microinches",
    9: "Mils",
    10: "Yards",
    11: "Angstroms",
    12: "Nanometers",
    13: "Microns",
    14: "Decimeters",
    15: "Decameters",
    16: "Hectometers",
    17: "Gigameters",
    18: "Astronomical units",
    19: "Light years",
    20: "Parsecs",
}
print("Drawing Units:", unit_map.get(units, "Unknown"))


class Wall:
    def __init__(self, start, end):
        self.start = tuple(start)
        self.end = tuple(end)
        self.geom = LineString([self.start, self.end])
        self._gross_length = self.geom.length
        self.net_length = self._gross_length
        self.doors = []
        self.windows = []

    @property
    def gross_length(self):
        return self._gross_length

    def add_door(self, door):
        self.doors.append(door)
        self.net_length -= door.width

    def add_window(self, window):
        self.windows.append(window)
        self.net_length -= window.width



class Door:
    def __init__(self, point, width=0.9, height=2.1, door_type="Single"):
        self.point = point
        self.width = width
        self.height = height
        self.door_type = door_type

class Window:
    def __init__(self, point, width=1.5, height=1.2, sill_height=0.9, glazing="Single"):
        self.point = point
        self.width = width
        self.height = height
        self.sill_height = sill_height
        self.glazing = glazing



class Room:
    def __init__(self, polygon, label=None):
        self.polygon = polygon
        self.label = label
        self.walls = []

    @property
    def area(self):
        return self.polygon.area

    @property
    def perimeter(self):
        return sum(w.gross_length for w in self.walls)

    @property
    def net_wall_length(self):
        return sum(w.net_length for w in self.walls)

  rooms = []
room_labels = {}

# --- Step 1: Collect room boundaries ---
for e in msp.query("LWPOLYLINE"):
    if e.closed:
        pts = [(p[0], p[1]) for p in e.get_points()]
        if len(pts) > 2:
            poly = Polygon(pts)
            room = Room(poly)
            rooms.append(room)

# --- Step 2: Collect text labels (assign names) ---
texts = []
for e in msp.query("TEXT MTEXT"):
    label = e.dxf.text if e.dxftype() == "TEXT" else e.text
    insert = (e.dxf.insert[0], e.dxf.insert[1])
    texts.append((label.strip(), Point(insert)))

for room in rooms:
    for label, pt in texts:
        if room.polygon.contains(pt):
            room.label = label
            break

def normalize_wall(p1, p2, tol=3):
    p1 = (round(p1[0], tol), round(p1[1], tol))
    p2 = (round(p2[0], tol), round(p2[1], tol))
    return tuple(sorted([p1, p2]))

walls_dict = {}  # avoid duplicates across rooms


for r_index, room in enumerate(rooms):
    coords = list(room.polygon.exterior.coords)

    for j in range(len(coords) - 1):
        p1, p2 = coords[j], coords[j+1]
        wkey = normalize_wall(p1, p2)

        if wkey not in walls_dict:
            walls_dict[wkey] = Wall(start=p1, end=p2)  # use explicit named args
        room.walls.append(walls_dict[wkey])

def snap_point(pt, grid=0.05):
    """Snap points to a grid (e.g., 1cm = 0.01)."""
    return (round(pt[0]/grid)*grid, round(pt[1]/grid)*grid)

def wall_key(p1, p2):
    """Consistent key for wall, ignoring direction."""
    return tuple(sorted([snap_point(p1), snap_point(p2)]))

min_length = 0.05  # Minimum wall length (e.g., 5cm)

walls_dict = {}
for room in rooms:
    coords = list(room.polygon.exterior.coords)
    for j in range(len(coords) - 1):
        p1, p2 = coords[j], coords[j+1]
        key = wall_key(p1, p2)
        line = LineString([snap_point(p1), snap_point(p2)])
        if key not in walls_dict and line.length >= min_length:
            walls_dict[key] = line
        # Always append wall geometry so room has access
        if key in walls_dict:
            room.walls.append(walls_dict[key])

# --- Plot results ---
fig, ax = plt.subplots(figsize=(6,6))
for line in walls_dict.values():
    x, y = line.xy
    ax.plot(x, y, 'k-', linewidth=2)
ax.set_aspect("equal")
ax.set_title(f"Unique walls preview (count={len(walls_dict)})")
plt.show()

def normalize_wall(p1, p2, tol=3):
    """Round coordinates and make wall representation order-independent"""
    p1 = (round(p1[0], tol), round(p1[1], tol))
    p2 = (round(p2[0], tol), round(p2[1], tol))
    return tuple(sorted([p1, p2]))

# Collect unique walls across all rooms
walls_dict = {}

for room in rooms:
    coords = list(room.polygon.exterior.coords)
    for j in range(len(coords) - 1):
        p1, p2 = coords[j], coords[j+1]
        wkey = normalize_wall(p1, p2)

        if wkey not in walls_dict:
            geom = LineString([p1, p2])
            walls_dict[wkey] = {
                "x1": p1[0],
                "y1": p1[1],
                "x2": p2[0],
                "y2": p2[1],
                "length": round(geom.length, 2)
            }

# Export clean walls
export_walls = list(walls_dict.values())


# ensure walls_dict exists and has consistent indexing for exporting
wall_keys = list(walls_dict.keys())
# map wall key -> integer id (1-based)
wall_id_map = {k: i+1 for i, k in enumerate(wall_keys)}

doors_found = []
windows_found = []

# simple parser for block names: try to read sizes like "door_900x2100" -> width mm, height mm
def parse_block_size(name):
    # returns (width_m, height_m) or (None, None)
    import re
    m = re.search(r'(\d{2,4})\s*[xX]\s*(\d{3,4})', name)
    if m:
        w = float(m.group(1))/1000.0
        h = float(m.group(2))/1000.0
        return w, h
    return None, None

# collect INSERTs
for e in msp.query("INSERT"):
    blk = e.dxf.name.lower()
    insert = (e.dxf.insert[0], e.dxf.insert[1])
    pt = Point(insert)

    # decide type
    is_door = "door" in blk or "puerta" in blk or blk.startswith("d")
    is_win  = "window" in blk or "vent" in blk or blk.startswith("w") or "win" in blk

    if not (is_door or is_win):
        continue

    # parse size if available
    w_mm, h_mm = parse_block_size(blk)
    if w_mm is None:
        # fallback defaults (meters)
        if is_door:
            width = 1.37
            height = 2.10
        else:
            width = 1.50
            height = 1.20
    else:
        width, height = w_mm, h_mm

    # find nearest unique wall (walls_dict values)
    if len(walls_dict) == 0:
        print("⚠️ No walls in walls_dict; skipping door/window assignment")
        continue

    # choose nearest wall by distance (but ensure projection falls on segment)
    best = None
    best_dist = float("inf")
    best_proj = None
    best_param = None  # fraction along line (0..1)
    for k, w in walls_dict.items():
        # w.geom is LineString
        proj = w.geom.interpolate(w.geom.project(pt))  # point on line
        dist = proj.distance(pt)
        # compute parameter t: project length / total length
        proj_len = w.geom.project(pt)
        t = proj_len / w.geom.length if w.geom.length > 0 else 0

        # ensure projection falls on segment (t between 0 and 1 with small tolerance)
        if t < -1e-6 or t > 1.000001:
            continue

        if dist < best_dist:
            best = (k, w)
            best_dist = dist
            best_proj = proj
            best_param = t

    # discard if too far from any wall (safety threshold, e.g. 0.5 m)
    if best is None or best_dist > 0.5:
        print(f"⚠️ Couldn't attach block {blk} at {insert} to any wall (dist {best_dist:.2f} m).")
        continue

    wall_key, wall_obj = best
    # compute angle of the wall for orientation (radians -> degrees)
    sx, sy = wall_obj.start
    ex, ey = wall_obj.end
    angle_rad = atan2(ey - sy, ex - sx)
    angle_deg = degrees(angle_rad)

    # compute insertion point coordinates (x,y)
    ipx, ipy = best_proj.x, best_proj.y

    if is_door:
        door = Door((ipx, ipy), width=width, height=height)
        # attach door to that wall object (adds to walls' door list and adjusts deduction)
        wall_obj.add_door(door)
        doors_found.append({
            "wall_id": wall_id_map[wall_key],
            "x": round(ipx, 4),
            "y": round(ipy, 4),
            "width": round(width, 3),
            "height": round(height, 3),
            "angle": round(angle_deg, 2)
        })
    else:
        win = Window((ipx, ipy), width=width, height=height)
        wall_obj.add_window(win)
        windows_found.append({
            "wall_id": wall_id_map[wall_key],
            "x": round(ipx, 4),
            "y": round(ipy, 4),
            "width": round(width, 3),
            "height": round(height, 3),
            "angle": round(angle_deg, 2)
        })

# Export to Excel (or JSON), put doors and windows on separate sheets

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
export_path = r"C:\Users\dhuvo\Desktop\Construction_Exports"  # change to your configured path
os.makedirs(export_path, exist_ok=True)
excel_file = os.path.join(export_path, f"attrexports_{timestamp}.xlsx")

with pd.ExcelWriter(excel_file) as writer:
    # unique walls sheet (for Dynamo walls)
    wall_rows = []
    for k, w in walls_dict.items():
        wall_rows.append({
            "wall_id": wall_id_map[k],
            "x1": w.start[0], "y1": w.start[1],
            "x2": w.end[0],   "y2": w.end[1],
            "length": round(w.gross_length, 3)
        })
    pd.DataFrame(wall_rows).to_excel(writer, sheet_name="walls", index=False)

    # doors sheet
    pd.DataFrame(doors_found).to_excel(writer, sheet_name="doors", index=False)

    # windows sheet
    pd.DataFrame(windows_found).to_excel(writer, sheet_name="windows", index=False)

print("✅ Exported walls/doors/windows to:", excel_file)

print(f"Total walls detected: {len(walls_dict)}")

for e in msp.query("INSERT"):
    block_name = e.dxf.name.lower()
    insert_point = (e.dxf.insert[0], e.dxf.insert[1])

    if "door" in block_name or "puerta" in block_name:
        door = Door(insert_point)
        # assign to nearest wall
        nearest = min(walls_dict.values(), key=lambda w: w.geom.distance(door.point))
        nearest.add_door(door)

    elif "window" in block_name or "vent" in block_name:
        window = Window(insert_point)
        nearest = min(walls_dict.values(), key=lambda w: w.geom.distance(window.point))
        nearest.add_window(window)

for i, room in enumerate(rooms, start=1):
    label = room.label or f"Unnamed_Room_{i}"
    print(f"{label}:")
    print(f"  Area = {room.area:.2f} m²")
    print(f"  Gross Perimeter = {room.perimeter:.2f} m")
    print(f"  Net Wall Length = {room.net_wall_length:.2f} m")
    for k, w in enumerate(room.walls, start=1):
        print(f"    Wall {k}: gross={w.gross_length:.2f} m, net={w.net_length:.2f} m "
              f"(doors={len(w.doors)}, windows={len(w.windows)})")

import matplotlib.pyplot as plt

# --- Plot rooms, walls, doors, windows ---
fig, ax = plt.subplots(figsize=(12, 10))

for i, room in enumerate(rooms):
    # Plot room polygon
    x, y = room.polygon.exterior.xy
    ax.fill(x, y, alpha=0.1, color="blue")
    ax.plot(x, y, color="blue", linewidth=1)
    
    # Label the room
    ax.text(room.polygon.centroid.x, room.polygon.centroid.y,
            room.label or f"Unnamed_Room_{i+1}",
            fontsize=12, color="blue", ha="center")
    # Fix net length so it never goes negative
    wall_net = max(0, room.net_wall_length)
    
    # Plot text
    ax.text(wall.geom.centroid.x, wall.geom.centroid.y,
            f"{wall_net:.2f}m", fontsize=7, color="black")
    
    # Add legend handles (only once)
    import matplotlib.lines as mlines
    door_line = mlines.Line2D([], [], color='green', linewidth=3, label='Door')
    window_line = mlines.Line2D([], [], color='red', linewidth=3, label='Window')
    wall_line = mlines.Line2D([], [], color='black', linewidth=2, label='Wall')
    
    ax.legend(handles=[wall_line, door_line, window_line], loc="upper right")


    # Plot walls
    for wall in room.walls:
        wall_net = max(0, wall.net_length)
        if wall_net > 0.05:  # only show if >5cm
            ax.text(wall.geom.centroid.x, wall.geom.centroid.y,
                    f"{wall_net:.2f}m", fontsize=7, color="black")


        # Plot doors as green line segments
        for d in wall.doors:
            mid = wall.geom.interpolate(wall.geom.project(d.point))
            offset = d.width / 2
            segment = LineString([
                wall.geom.interpolate(wall.geom.project(d.point) - offset),
                wall.geom.interpolate(wall.geom.project(d.point) + offset)
            ])
            dx, dy = segment.xy
            ax.plot(dx, dy, color="green", linewidth=3, label="Door")

        # Plot windows as red line segments
        for w in wall.windows:
            mid = wall.geom.interpolate(wall.geom.project(w.point))
            offset = w.width / 2
            segment = LineString([
                wall.geom.interpolate(wall.geom.project(w.point) - offset),
                wall.geom.interpolate(wall.geom.project(w.point) + offset)
            ])
            wx, wy = segment.xy
            ax.plot(wx, wy, color="red", linewidth=3, label="Window")

ax.set_aspect("equal", "box")
plt.title("Rooms, Walls, Doors & Windows")
plt.show()

def clean_walls(walls, min_length=0.1, tol=1e-3):

    def canonical(wall):
        """Return canonicalized (sorted & rounded) endpoints."""
        coords = list(wall.geom.coords)
        p1 = (round(coords[0][0]/tol)*tol, round(coords[0][1]/tol)*tol)
        p2 = (round(coords[1][0]/tol)*tol, round(coords[1][1]/tol)*tol)
        return tuple(sorted([p1, p2]))

    seen = set()
    cleaned = []

    for wall in walls:
        length = wall.geom.length
        if length < min_length:
            continue  # skip tiny wall

        key = canonical(wall)
        if key in seen:
            continue  # skip duplicate
        seen.add(key)

        cleaned.append(wall)

    return cleaned


for room in rooms:
    room.walls = clean_walls(room.walls, min_length=0.2, tol=1e-3)

export_data = {"rooms": []}
rows = []

export_walls = []
for wall in walls_dict.values():
    export_walls.append({
        "x1": wall.start[0],
        "y1": wall.start[1],
        "x2": wall.end[0],
        "y2": wall.end[1],
        "length": wall.gross_length
    })


for room in rooms:  
    room_entry = {
        "name": room.label,
        "area": round(room.polygon.area, 2),
        "walls": []
    }

    for i, wall in enumerate(room.walls, start=1):
        wall_entry = {
            "id": i,
            "gross_length": round(wall.gross_length, 2),
            "net_length": round(wall.net_length, 2),
            "doors": [{"x": d.point.x, "y": d.point.y, "width": d.width} for d in wall.doors],
            "windows": [{"x": w.point.x, "y": w.point.y, "width": w.width} for w in wall.windows],
        }
        room_entry["walls"].append(wall_entry)

        # flat record for Excel
        rows.append({
            "Room": room.label,
            "Room_Area": round(room.polygon.area, 2),
            "Wall_ID": i,
            "Gross_Length": round(wall.gross_length, 2),
            "Net_Length": round(wall.net_length, 2),
            "Doors": len(wall.doors),
            "Windows": len(wall.windows),
             # new: wall endpoints
            "x1": round(wall.geom.coords[0][0], 3),
            "y1": round(wall.geom.coords[0][1], 3),
            "x2": round(wall.geom.coords[1][0], 3),
            "y2": round(wall.geom.coords[1][1], 3),           
        })

    export_data["rooms"].append(room_entry)



# --- Debug plot of exported walls ---
fig, ax = plt.subplots(figsize=(8, 8))

for i, wall in enumerate(export_walls, start=1):
    x = [wall["x1"], wall["x2"]]
    y = [wall["y1"], wall["y2"]]
    ax.plot(x, y, color="black", linewidth=2)
    # add wall ID for cross-check
    midx = (wall["x1"] + wall["x2"]) / 2
    midy = (wall["y1"] + wall["y2"]) / 2
    ax.text(midx, midy, f"W{i}", fontsize=8, color="red")

# Label rooms too
for i, room in enumerate(rooms, start=1):
    cx, cy = room.polygon.centroid.x, room.polygon.centroid.y
    ax.text(cx, cy, room.label or f"Room {i}", fontsize=8, color="blue", ha="center")

ax.set_aspect("equal", "box")
plt.show()

# --- CONFIGURATION ---
EXPORT_FOLDER = os.path.join(os.path.expanduser("~"), "Desktop", "Construction_Exports")

os.makedirs(EXPORT_FOLDER, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

json_file = os.path.join(EXPORT_FOLDER, f"rooms_export_{timestamp}.json")
excel_file = os.path.join(EXPORT_FOLDER, f"rooms_export_{timestamp}.xlsx")

with open(json_file, "w") as f:
    json.dump(export_data, f, indent=4)

df = pd.DataFrame(export_walls)
df.to_excel(excel_file, index=False)

print(f"✅ Exported JSON to {json_file}")
print(f"✅ Exported Excel to {excel_file}")
