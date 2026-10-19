"""
Verdant 3D - Roblox-native multi-AI backend (v18 - Cloud 3D Smoothing, Rigging & Facial Relief Engine)

Upgrades in v18:
1. Facial Depth Relief & Feature Extruder (extrude_facial_hierarchy):
   Mathematically guarantees the creature's face is NEVER buried inside the head/muzzle!
   Automatically detects the head bounding box and steps facial features forward along +Z
   in 4 distinct relief layers: Cranium -> Faceplate/Muzzle/Jaw -> Mouth Cavity/Glowing Eyes/Brows
   -> Pupils/White Fangs/Pink Tongue.
2. Cloud 3D Organic Smoothing Engine (apply_organic_smoothing):
   Replaces razor-sharp 0.04 cube corners with adaptive sculptural beveling (smooth 0.25-0.48)
   on organic muscles, heads, cheeks, chests, and limbs, while keeping crowns, blades,
   teeth, and claws crisp.
3. Automatic Rigging Joint Welder (smooth_and_weld_rigging):
   Eliminates sketchy, jagged staircase arms and legs by aligning limb vectors, tapering
   consecutive segments smoothly, and welding inter-segment seams.
4. Fixed Word-Boundary Color Blocking:
   Prevents 'Torso_Core' or 'Hand_Core' from accidentally matching 'core' (red accent),
   keeping fur deep navy-charcoal (#18202c), chest/face/knuckles/toes warm cream (#e8dcc4),
   wrist cuffs & crown royal gold (#f5c518), and eyes/fangs/tongue crisp.
"""
import asyncio
import colorsys
import json
import math
import os
import re
import time
import zlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Verdant 3D - Cloud 3D Modeling & Smoothing Backend (v18)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --------------------------------------------------------------------------- config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "gemini-3.8-flash")
THINKING_LEVEL = os.getenv("THINKING_LEVEL", "medium").lower().strip()

DEFAULT_MODELS = {
    "gemini": AI_MODEL,
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
}
ENV_KEYS = {
    "gemini": GEMINI_API_KEY,
    "anthropic": ANTHROPIC_API_KEY,
    "openai": OPENAI_API_KEY,
    "groq": GROQ_API_KEY,
}

AUTO_REVISE = os.getenv("AUTO_REVISE", "1") != "0"
try:
    MAX_REVISIONS = max(0, min(2, int(os.getenv("MAX_REVISIONS", "1"))))
except ValueError:
    MAX_REVISIONS = 1

REVISE_BUDGET_S = 95.0
RETRY_STATUS = (429, 500, 502, 503, 504, 529)

GEMINI_MODEL_ALIASES = {
    "argon": "gemini-4-argon",
    "argon-medium": "gemini-4-argon",
    "gemini-argon": "gemini-4-argon",
    "gemini-4-argon": "gemini-4-argon",
    "gemini-4-argon-medium": "gemini-4-argon",
    "3.8flash": "gemini-3.8-flash",
    "3.8-flash": "gemini-3.8-flash",
    "gemini-3.8-flash": "gemini-3.8-flash",
    "3.7flash": "gemini-3.7-flash",
    "3.7-flash": "gemini-3.7-flash",
    "gemini-3.7-flash": "gemini-3.7-flash",
    "3.6flash": "gemini-3.6-flash",
    "3.6-flash": "gemini-3.6-flash",
    "gemini-3.6-flash": "gemini-3.6-flash",
    "3.5-flash": "gemini-3.5-flash",
    "gemini-3.5-flash-lite": "gemini-3.5-flash-lite",
    "gemini-2.5-flash": "gemini-3.8-flash",
    "gemini-2.5-pro": "gemini-3.8-flash",
}

THINKING_BUDGETS = {
    "none": 0,
    "off": 0,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
}

STUD = 0.28               # metres per Roblox stud
MAX_PARTS = 175
MAX_EXTENT_STUDS = 70.0
MAX_PART_STUDS = 28.0
MIN_PART_STUDS = 0.16
SCENE_MAX_M = 12.0
PUSH = 0.12               # 0.12 stud offset so decals/trim never z-fight
SHAPES = {"box", "sphere", "cylinder", "cone", "torus"}
SMOOTH_DEFAULT = {"box": 0.22, "sphere": 0.65, "cylinder": 0.35, "cone": 0.25, "torus": 0.45}
SHAPE_ALIAS = {
    "cube": "box", "block": "box", "brick": "box", "plate": "box", "wedge": "box", "ramp": "box",
    "slope": "box", "part": "box", "roof": "box", "panel": "box", "plank": "box",
    "ball": "sphere", "orb": "sphere", "ellipsoid": "sphere", "dome": "sphere", "drop": "sphere",
    "lobe": "sphere", "capsule": "sphere", "ovoid": "sphere",
    "tube": "cylinder", "pillar": "cylinder", "rod": "cylinder", "pipe": "cylinder",
    "column": "cylinder", "disc": "cylinder", "disk": "cylinder", "wheel": "cylinder",
    "tire": "cylinder", "tyre": "cylinder", "barrel": "cylinder", "shaft": "cylinder",
    "pyramid": "cone", "spike": "cone", "horn": "cone", "fang": "cone", "tooth": "cone",
    "ring": "torus", "donut": "torus", "hoop": "torus", "collar": "torus",
}
PART_RANGE = {
    "creature": (98, 142), "humanoid": (92, 134), "vehicle": (88, 128),
    "structure": (92, 138), "prop": (64, 104), "plant": (74, 116),
}
GATE = 0.80

# --------------------------------------------------------------------------- the skill
CORE_SKILL = """You are the VERDANT CLOUD 3D SCULPTOR: an elite Roblox Studio character & boss modeler. You build smooth, muscular, organically beveled, richly detailed 3D models that match front-page Roblox game bosses (like the Gorilla King Boss with a sculpted cream face, fierce red eyes under angled brows, open roaring mouth with white fangs & pink tongue, golden crown, layered dark shoulder fur, sculpted cream V-chest & abs, golden wrist cuffs, and thick muscular limbs).

OUTPUT - reply with ONE valid JSON object and nothing else (no markdown fences, no comments, no prose):
{
  "title": "Iconic Game-Ready Title",
  "category": "creature|humanoid|vehicle|structure|prop|plant",
  "plan": {
    "reference": "2-3 sentences: exact anatomy, smooth muscular rigging flow, facial expression, and color zones",
    "must_have": ["countable features with explicit colors, e.g. 'protruding cream face & roaring jaw with 6 white fangs & tongue', 'glowing red eyes under angled brow ridge', '5-point golden crown', 'cream sculpted chest & abs', 'gold wrist bracers', 'layered dark shoulder fur pads'"],
    "signature_features": ["6-10 traits that give this model front-page Roblox game appeal"],
    "palette": {
      "base": "#18202c",
      "secondary": "#e8dcc4",
      "trim": "#f5c518",
      "accent": "#ff2a2a",
      "dark": "#0d1017",
      "light": "#ffffff"
    },
    "proportions": "one sentence with exact width:height:length ratios in studs",
    "sections": {"Head_And_Face": 28, "Torso_And_Chest": 24, "Arms_And_Hands": 30, "Legs_And_Feet": 18, "Crown_And_Armor": 14}
  },
  "ops": [
    {"id":"Group_Part","shape":"box|cylinder|sphere|cone|torus","size":[w,h,d],"pos":[x,y,z],"rot":[degX,degY,degZ],"color":"#rrggbb","smooth":0.32}
  ]
}

CLOUD 3D SCULPTING, RIGGING & FACIAL LAWS (FOLLOW STRICTLY):
1. COORDINATES: Units are STUDS. Y is UP, ground is y=0, model is centred on x=0, z=0, and the FRONT faces +Z (X = left/right width, Y = vertical height, Z = front/back length).
2. ORGANIC BEVEL SMOOTHING (`smooth` property):
   - Use `"smooth": 0.30` to `0.45` on all organic body masses, muscles, heads, cheeks, chests, bellies, shoulders, arms, legs, and fingers so they render as smooth, sculpted Roblox meshes rather than sharp-edged cubes!
   - Use `"smooth": 0.04` to `0.10` ONLY on sharp geometric parts: teeth, fangs, claws, crown spikes, blades, and stone walls.
3. SMOOTH RIGGING & MUSCULAR FLOW (NO SKETCHY STAIRCASE LIMBS!):
   - Every arm and leg must flow along a smooth anatomical curve with overlapping spherical/rounded joints (`Shoulder_Joint`, `Elbow_Joint`, `Wrist_Joint`, `Hip_Joint`, `Knee_Joint`).
   - Do NOT stack sideways offset blocks like a broken staircase! Keep the X coordinate of UpperArm, Elbow, Forearm, Bracer, and Hand aligned within 0.5 studs of each other, curving gently forward in +Z (`z = 2.0 -> 3.2 -> 4.5 -> 5.2`).
4. 3D FACIAL RELIEF HIERARCHY (THE FACE MUST POP OUT CLEARLY AT +Z!):
   - Never bury the eyes, brows, or mouth inside the head block! Build the head in 4 forward-stepping Z layers:
     * Layer 1 (z = 3.5..5.5): `Head_Cranium`, `Head_Crest`, `Head_Cheek_L`, `Head_Cheek_R` in `base` dark fur (`#18202c`).
     * Layer 2 (z = 7.2..8.2): `Head_FacePlate` and `Head_UpperSnout` in `secondary` cream (`#e8dcc4`), plus `Head_Jaw_Lower` below at y=16.2, z=8.2.
     * Layer 3 (z = 8.8..9.2): Heavy angled `Head_Brow_L` & `Head_Brow_R` (`rot:[0,0,-14]` / `[0,0,14]`) overhanging fierce glowing `Eye_L` & `Eye_R` (`accent` `#ff2a2a`), plus a deep dark `Head_Mouth_Cavity` (`#0d1017`) between the upper snout and lower jaw!
     * Layer 4 (z = 9.4..9.8): Dark `Pupil_L` & `Pupil_R`, 2 nostrils `Head_Nostril_L`/`R`, pink `Head_Tongue` (`#e06b80`), and 6 crisp white `Fang_UL`, `Fang_UR`, `Tooth_U1`, `Tooth_U2`, `Fang_LL`, `Fang_LR` (`#ffffff`) clearly visible inside the open roaring mouth!
5. STRICT 6-ZONE COLOR PALETTE (NEVER PUT RED ON CHEST OR HANDS!):
   - `base` (`#18202c`): Dark navy-charcoal fur on Cranium, Back, Torso sides, Shoulders, Biceps, Forearms, Thighs, Shins, Hand/Foot backs.
   - `secondary` (`#e8dcc4`): Warm cream/tan on `Head_FacePlate`, `Head_Snout`, `Head_Jaw_Lower`, sculpted `Chest_Pec_L`, `Chest_Pec_R`, `Belly_Ab1..4` (forming an inverted V/heart chest plate), plus the 8 knuckle fingers (`Finger_L_1..4`, `Finger_R_1..4`) and 8 toes (`Toe_L_1..4`, `Toe_R_1..4`).
   - `trim` (`#f5c518`): Bright royal gold ONLY on the `Crown_*` on top of the head and the thick ribbed `Wrist_Bracer_L_*` / `Wrist_Bracer_R_*` cuffs around the forearms! Do NOT put a gold belt on a gorilla's waist!
   - `accent` (`#ff2a2a`): Glowing red ONLY on `Eye_L` and `Eye_R` (never on torso, waist, or hands!).
   - `light` (`#ffffff`): Pure white on teeth, fangs, and claws.
   - `dark` (`#0d1017`): Obsidian on `Head_Mouth_Cavity`, `Pupil_*`, and `Head_Nostril_*`."""

BLUEPRINTS = {
    "creature_quad": """BLUEPRINT - STUDIO BOSS APE / GORILLA / QUADRUPED / DINO / BEAST, @LO@-@HI@ parts, with smooth organic rigging and high-relief face:
- IF GORILLA / APE / KING KONG / YETI / TROLL:
  - Follow this EXACT ANATOMICAL TEMPLATE (front = +Z, ground y=0, smooth=0.35 on all body/limb/face parts):
  - 1. HEAD & 4-LAYER ROARING FACE (28-34 parts, centered at x=0, y=17..25):
    - Back Skull & Fur: `Head_Cranium` box [8.0, 7.2, 6.5] at [0, 20.5, 4.2] (`base`), `Head_TopDome` [6.5, 2.2, 5.5] at [0, 24.2, 4.0] (`base`), thick side fur `Head_Cheek_L` / `Head_Cheek_R` [2.4, 5.2, 5.2] at [-4.4, 19.8, 4.8] / [4.4, 19.8, 4.8] (`base`), `Head_Ear_L` / `Head_Ear_R` [1.2, 2.0, 1.6] at [-5.2, 21.0, 4.5] / [5.2, 21.0, 4.5] (`secondary`).
    - Cream Face & Snout (Layer 2, z=7.6..8.4, `secondary` `#e8dcc4`): `Head_FaceUpper` [6.4, 3.6, 1.8] at [0, 21.4, 7.5], `Head_CheekBone_L` / `R` [2.2, 3.2, 2.0] at [-2.4, 19.4, 7.8] / [2.4, 19.4, 7.8], `Head_Muzzle_Upper` [5.2, 1.8, 2.4] at [0, 19.5, 8.4], `Head_Jaw_Lower` [4.8, 1.5, 2.6] at [0, 16.3, 8.3], `Head_Chin_Fur` [5.2, 1.2, 2.4] at [0, 15.3, 7.8] (`base`).
    - Brows, Glowing Red Eyes & Open Mouth Cavity (Layer 3, z=8.8..9.3):
      * Heavy Cream Brow Ridge: `Head_Brow_Center` [2.4, 1.1, 1.8] at [0, 22.1, 8.8] (`secondary`), `Head_Brow_L` [2.8, 1.1, 1.8] at [-1.9, 22.3, 8.8] rot [0,0,-14] (`secondary`), `Head_Brow_R` [2.8, 1.1, 1.8] at [1.9, 22.3, 8.8] rot [0,0,14] (`secondary`).
      * Glowing Red Eyes: `Eye_L` / `Eye_R` [1.8, 0.95, 0.6] at [-1.9, 21.2, 8.9] / [1.9, 21.2, 8.9] (`accent` `#ff2a2a`), `Pupil_L` / `Pupil_R` [0.65, 0.85, 0.4] at [-1.8, 21.2, 9.25] / [1.8, 21.2, 9.25] (`dark`).
      * Nostrils: `Head_Nostril_L` / `R` [0.6, 0.7, 0.5] at [-0.8, 19.8, 9.65] / [0.8, 19.8, 9.65] (`dark`).
      * Roaring Mouth Interior: `Head_Mouth_Cavity` [4.0, 2.2, 1.4] at [0, 17.9, 8.6] (`dark` `#0d1017`), `Head_Tongue` [2.8, 0.7, 1.8] at [0, 17.2, 8.9] (`#d95b72`).
      * 6 White Fangs & Teeth (`light` `#ffffff`, smooth=0.05): `Fang_UL` / `Fang_UR` [0.65, 1.1, 0.65] at [-1.5, 18.4, 9.4] / [1.5, 18.4, 9.4], `Tooth_U1` / `Tooth_U2` [0.55, 0.7, 0.55] at [-0.5, 18.6, 9.45] / [0.5, 18.6, 9.45], `Fang_LL` / `Fang_LR` [0.6, 1.0, 0.6] at [-1.4, 17.2, 9.4] / [1.4, 17.2, 9.4].
    - Golden Royal Crown (9 parts at y=24.8..28.2, `trim` `#f5c518`, smooth=0.12):
      * `Crown_Base` cylinder [5.8, 1.4, 5.8] at [0, 25.0, 4.0], `Crown_InnerDome` sphere [4.4, 2.2, 4.4] at [0, 25.6, 4.0] (`base`), `Crown_Rim` box [6.0, 0.8, 6.0] at [0, 25.6, 4.0], `Crown_Spike_Front` [1.6, 2.8, 1.2] at [0, 27.0, 6.6], `Crown_Spike_FL` / `FR` [1.3, 2.3, 1.2] at [-2.2, 26.7, 5.8] / [2.2, 26.7, 5.8] rot [0,0,15] / [0,0,-15], `Crown_Spike_L` / `R` [1.2, 2.1, 1.2] at [-2.8, 26.5, 4.0] / [2.8, 26.5, 4.0], `Crown_Gem` [1.0, 1.2, 0.6] at [0, 25.8, 7.0] (`#ffdf4d`).
  - 2. MUSCULAR TORSO, LAYERED SHOULDER FUR & SCULPTED CREAM CHEST (24-28 parts):
    - Core Body (`base` `#18202c`, smooth=0.38): `Torso_Upper` [14.5, 8.5, 9.5] at [0, 15.8, 1.5], `Torso_Mid` [12.0, 6.0, 8.2] at [0, 11.2, 0.8], `Torso_Pelvis` [10.5, 4.8, 7.5] at [0, 7.5, 0.0], `Back_Hump` [11.5, 5.5, 4.0] at [0, 17.5, -2.5].
    - Sculpted Cream Chest & Tapered Abs (`secondary` `#e8dcc4`, smooth=0.30, proud on front +Z — NO RED OR GOLD ON TORSO!):
      * `Chest_Pec_L` / `Chest_Pec_R` [5.2, 4.4, 1.8] at [-2.8, 16.0, 6.2] / [2.8, 16.0, 6.2], `Chest_Bridge` [3.6, 3.8, 1.7] at [0, 15.8, 6.3], `Belly_Upper` [7.6, 2.8, 1.6] at [0, 13.0, 5.2], `Belly_Mid` [5.6, 2.4, 1.5] at [0, 10.8, 4.6], `Belly_LowerPoint` [3.4, 2.0, 1.4] at [0, 9.0, 4.0].
    - Layered Jagged Shoulder Fur Pads (`base` dark slate, 5 per side):
      * `Pauldron_L_1` [6.5, 2.2, 7.5] at [-8.2, 20.2, 2.2] rot [0,0,15], `Pauldron_L_2` [7.0, 2.2, 7.8] at [-9.2, 18.8, 2.2] rot [0,0,22], `Pauldron_L_3` [6.8, 2.0, 7.2] at [-9.8, 17.2, 2.2] rot [0,0,28], plus matching `Pauldron_R_1..3` on +X!
  - 3. SMOOTH RIGGED ARMS, GOLDEN WRIST CUFFS & CREAM KNUCKLE HANDS (30-36 parts):
    - Keep Left Arm aligned along X = -9.4..-10.2 and Right Arm along X = +9.4..+10.2 (NO jagged sideways steps!):
    - Left Arm Chain (`base` `#18202c`, smooth=0.40): `Arm_L_Shoulder` sphere [6.8, 6.8, 6.8] at [-9.0, 17.5, 2.2], `Arm_L_Bicep` box [6.2, 6.2, 6.2] at [-9.6, 13.8, 3.0] rot [12,0,6], `Arm_L_Elbow` sphere [5.8, 5.2, 5.8] at [-9.8, 10.6, 3.8], `Arm_L_Forearm` box [6.8, 6.4, 6.8] at [-9.8, 7.6, 4.6] rot [10,0,0].
    - Left Golden Wrist Bracer (`trim` `#f5c518`, smooth=0.25): `Bracer_L_Upper` [7.3, 1.1, 7.3] at [-9.8, 8.8, 4.6] rot [10,0,0], `Bracer_L_Main` [7.2, 2.4, 7.2] at [-9.8, 7.2, 4.8] rot [10,0,0], `Bracer_L_Lower` [7.3, 1.1, 7.3] at [-9.8, 5.6, 5.0] rot [10,0,0].
    - Left Hand & Cream Knuckles (at y=0..4.4): `Hand_L_Back` (`base` `#18202c`, NEVER RED!) [6.8, 3.2, 5.8] at [-9.8, 3.2, 5.4], plus 4 cream/tan `secondary` (`#e8dcc4`) rounded knuckle fingers on the front: `Finger_L_1` [1.5, 2.6, 2.4] at [-12.1, 1.3, 7.4], `Finger_L_2` [1.5, 2.8, 2.5] at [-10.5, 1.4, 7.6], `Finger_L_3` [1.5, 2.8, 2.5] at [-8.9, 1.4, 7.6], `Finger_L_4` [1.5, 2.5, 2.3] at [-7.3, 1.2, 7.3], and `Thumb_L` [1.6, 2.2, 2.2] at [-6.8, 1.8, 5.6].
    - Mirror the exact same chain for Right Arm (`Arm_R_*`, `Bracer_R_*`, `Hand_R_Back`, `Finger_R_1..4`, `Thumb_R`) at X = +9.0..+9.8!
  - 4. SMOOTH RIGGED HIND LEGS & CREAM TOES (18-22 parts):
    - Left Leg (`base` `#18202c`, smooth=0.38): `Leg_L_Hip` sphere [5.6, 5.4, 5.6] at [-4.6, 7.2, -0.2], `Leg_L_Thigh` box [5.4, 5.2, 5.6] at [-4.8, 5.0, 0.2] rot [-12,0,0], `Leg_L_Shin` box [4.8, 4.4, 5.0] at [-4.8, 2.6, 0.0], `Foot_L_Ankle` (`base`) [5.2, 2.2, 5.4] at [-4.8, 1.1, 0.8], plus 4 large cream `secondary` (`#e8dcc4`) toes in front: `Toe_L_1..4` [1.25, 1.8, 2.4] across x=[-6.6, -5.4, -4.2, -3.0] at y=0.9, z=3.2 (and mirror for Right Leg `Leg_R_*`, `Foot_R_Ankle`, `Toe_R_1..4`).
- IF OTHER QUADRUPED / WOLF / LION / BEAR / DINO: use the same 4-layer protruding face relief (muzzle, open mouth cavity, fangs, tongue, glowing eyes under brows), smooth joint spheres (`smooth: 0.35`), and 6-zone color blocking.""",

    "creature_multi": """BLUEPRINT - CEPHALOPOD (SQUID / OCTOPUS / KRAKEN) OR ARACHNID / CRUSTACEAN, @LO@-@HI@ parts, organically smoothed (`smooth: 0.32`):
- IF SQUID / OCTOPUS / KRAKEN / CUTTLEFISH:
  - HEAD & PROTRUDING FACE: `Head_Core` [12,7,10] at [0,7.5,2]. `Head_Brow_L` / `Head_Brow_R` [4.5,1.2,1.8] at [-4,11.2,7.6] / [4,11.2,7.6]. `Head_Collar` [13,1.4,11] at [0,11.7,1.5] in `trim` or `secondary`.
  - BIG GLOWING EYES (proud at z=7.8..8.4): `Eye_L` / `Eye_R` [3.4,3.4,1.0] at [-4,8.6,7.8] / [4,8.6,7.8] in bright `accent`, `Pupil_L` / `Pupil_R` dark [0.9,2.6,0.5] at z=8.4.
  - BEAK & MOUTH: `Beak_Upper` [2.4,1.6,2.2] at [0,6.4,7.8], `Beak_Lower` [1.8,1.2,1.8] at [0,5.0,7.6] surrounded by 4 white tooth blocks.
  - FAT STEPPED MANTLE: 7 smooth-beveled boxes `Mantle_1` .. `Mantle_7`, each 3.2 tall, stepping up and back with 4 `secondary` cream belly plates underneath and 2 wide lateral fins.
  - 8 RADIAL ARMS + 2 FEEDING TENTACLES: 4 segments each fanning out at 45-degree yaw steps with smooth joint overlap and `light` sucker blocks underneath.""",

    "creature_dragon": """BLUEPRINT - DRAGON / WYVERN / WINGED BOSS, @LO@-@HI@ parts, organically beveled (`smooth: 0.30`):
- SILHOUETTE: wingspan 30-40 studs (X), length 28-36 studs (Z), proud boss stance.
- 4-LAYER HEAD, ROARING JAWS & HORNS (24-30): `Head_Cranium` at z=6, protruding `Head_Snout` & hinged lower `Head_Jaw` at z=9.5, dark `Head_Mouth_Cavity` & pink/fiery `Head_Tongue` at z=9.2, 8 white `Fang_*`/`Tooth_*` blocks at z=10.4, angled brow blocks, glowing `Eye_L`/`R` + pupils, 2 swept 4-segment horns (`Horn_L_1..4`, `Horn_R_1..4`) in `trim`/`light`, and a 4-segment neck chain `Neck_Seg1..4` with `secondary` throat scutes.
- TORSO (16-20): broad `Torso_Chest` in `base`, 6 overlapping `secondary` (cream/gold) belly scute plates (`Belly_Scute1..6`), haunches, and dorsal ridge spikes.
- WINGS (2 x 15-18): shoulder armor, upper/forearm wing bones, 4 fanning wing spars in `dark`/`trim`, and contrasting `secondary`/`accent` membrane plates.
- LEGS & TAIL (26-32): 4 muscular 4-segment legs with spherical knee/ankle joints, golden leg bands (`Bracer_*`), and 3 white `Claw_*` blocks per foot at y=0, plus a 7-segment tapering tail.""",

    "creature_bird": """BLUEPRINT - BIRD / PHOENIX / GRIFFIN BOSS, @LO@-@HI@ parts (`smooth: 0.30`):
- Stepped `base` body with 5 layered `secondary` chest plumage plates, 2-part hooked golden `trim` beak (`Beak_Upper`, `Beak_Lower`) protruding at +Z, glowing `accent` eyes + brow ridges, 6-part royal head crest/crown, 2 wide 3-layer feathered wings, fanned tail feathers, and 2 golden-taloned legs gripping a rocky pedestal.""",

    "creature_aquatic": """BLUEPRINT - AQUATIC / SHARK / LEVIATHAN BOSS, @LO@-@HI@ parts (`smooth: 0.35`):
- 9-segment tapering body along Z (`Body_Seg1..9`) with two-tone counter-shading (`base` dorsal top, bright `secondary` cream underbelly `Belly_Seg1..7`), open jaws at +Z lined with 10 white `Tooth_*` blocks and `Mouth_Cavity`, glowing `accent` eyes, gill slits, tall dorsal fin, pectoral fins, and twin tail flukes.""",

    "humanoid": """BLUEPRINT - HUMANOID / MECH / KNIGHT / BOSS CHARACTER, @LO@-@HI@ parts, 18-24 studs tall (`smooth: 0.25`):
- HEAD & HELMET/CROWN (18-24): `Head_Skull`, protruding `Face_Plate` (`secondary` skin or dark visor), glowing `Eye_L`/`Eye_R` + brows/mouth proud at +Z, and a multi-part helmet, hood, hair, or golden crown (`Crown_*`).
- TORSO & ARMOR (20-26): broad chest armor with contrasting `secondary` chest plates, glowing `accent` chest emblem, `trim` gold belt with buckle, and layered shoulder pauldrons (`Pauldron_L_1..3`, `Pauldron_R_1..3`).
- ARMS, GAUNTLETS & WEAPON (24-30): smoothly aligned `Shoulder -> Bicep -> Elbow -> Forearm -> Gold Bracer -> Hand + Fingers`, wielding a multi-part signature weapon.
- LEGS & BOOTS (18-22): `Hip -> Thigh -> KneeGuard (trim) -> Shin -> Boot + Toes` planted at y=0.""",

    "vehicle": """BLUEPRINT - VEHICLE (SUPERCAR / TRUCK / AIRCRAFT / SHIP), @LO@-@HI@ parts (`smooth: 0.18`):
- Multi-tone bodywork: `base` primary paint, `secondary` racing stripes/lower trim, `dark` carbon splitters/diffuser/tires, `trim` gold/silver wheel rims & badging, `light` bright headlights & window glass highlights, and `accent` glowing taillights/afterburners.
- Wheeled vehicles include 4 multi-part cylindrical wheels (`Tire`, `Rim`, `Hub`, `Caliper`), aerodynamic sloped hood/windshield/roof, side intakes, mirrors, quad exhausts, and rear spoiler wing.""",

    "structure": """BLUEPRINT - ARCHITECTURAL STRUCTURE / CASTLE / TOWER / SHRINE, @LO@-@HI@ parts (`smooth: 0.06`):
- Multi-tone materials: `dark`/`secondary` stepped stone foundation and corner quoins, `base` recessed wall panels, `trim` ornate roof trims/gold finials/banners, `accent` warm glowing window panes & hanging lanterns, `light` steps and arch keystones, plus pitched roof slopes and courtyard props.""",

    "prop": """BLUEPRINT - DETAILED PROP / WEAPON / ARTIFACT / CHEST, @LO@-@HI@ parts (`smooth: 0.15`):
- Crisp multi-material layering using all 6 palette slots: `dark` grip/iron frame, `trim` ornate gold crossguard/brackets/locks, `base` casing/scabbard, `secondary` blade fuller/inlay, `light` razor edges/rivets, and `accent` glowing runes/gems.""",

    "plant": """BLUEPRINT - STYLIZED TREE / BOTANICAL DIORAMA / FANTASY FLORA, @LO@-@HI@ parts (`smooth: 0.32`):
- Stepped earth & mossy rock base (`dark` + `secondary`), twisted multi-branch root & trunk chains (`base`), and 25-35 layered canopy/blossom/mushroom blocks across 3 distinct shades (`secondary`, `trim`, `accent`, `light`) with glowing fruit/spores.""",
}

WINGS_BP = """WINGS ADD-ON: Build 2 large symmetrical wings spreading wide along -X and +X from block shoulders. Each wing has a shoulder block, a 3-block leading-edge arm in `base`/`trim`, and 2-3 rows of 6-8 flat feather/membrane blocks in `secondary`/`accent` fanned out in 15 degree steps."""

KW = {
    "vehicle": r"\b(car|supercar|sportcar|sports car|hypercar|sedan|coupe|convertible|lamborghini|ferrari|porsche|bugatti|mclaren|mustang|camaro|corvette|gtr|supra|f1|formula|racecar|truck|pickup|lorry|bus|van|suv|jeep|hummer|limo|taxi|police car|ambulance|firetruck|tractor|motorcycle|motorbike|bike|bicycle|scooter|moped|atv|quad|tank|apc|humvee|train|locomotive|tram|subway|monorail|plane|airplane|biplane|airliner|fighter|jet|bomber|helicopter|chopper|drone|blimp|zeppelin|airship|rocket|spaceship|starship|spacecraft|shuttle|ufo|saucer|rover|lander|boat|ship|sailboat|yacht|speedboat|ferry|cruiser|battleship|carrier|submarine|sub|pirate ship|galleon|canoe|kayak|raft|jetski|hovercraft|sled|sleigh|chariot|wagon|cart|carriage|coach|rickshaw|kart|go-kart|buggy|forklift|bulldozer|excavator|crane|dump truck|cement mixer|combine|harvester|snowplow|snowmobile|segway|skateboard|hoverboard|rollercoaster)\b",
    "structure": r"\b(house|home|cottage|cabin|chalet|bungalow|villa|mansion|manor|estate|palace|chateau|castle|keep|citadel|fortress|fort|bastion|bunker|outpost|watchtower|guardtower|tower|turret|spire|obelisk|monument|memorial|arch|gate|gateway|portals?|wall|rampart|bridge|viaduct|aqueduct|overpass|pier|dock|wharf|harbor|lighthouse|beacon|windmill|watermill|mill|barn|silo|stable|coop|greenhouse|conservatory|shed|shack|hut|hovel|igloo|yurt|teepee|tent|pavilion|gazebo|pergola|pagoda|shrine|temple|sanctuary|monastery|abbey|church|cathedral|chapel|mosque|synagogue|stupa|pyramid|ziggurat|parthenon|colosseum|amphitheater|arena|stadium|coliseum|gym|pool|rink|shop|store|market|bazaar|stall|booth|kiosk|bakery|cafe|coffee shop|restaurant|diner|pizzeria|bar|pub|tavern|inn|hotel|motel|hostel|bank|vault|post office|library|school|university|college|academy|museum|gallery|theater|theatre|cinema|opera|studio|workshop|forge|smithy|foundry|factory|warehouse|depot|hangar|garage|gas station|service station|fire station|police station|hospital|clinic|pharmacy|prison|jail|dungeon|asylum|lab|laboratory|observatory|planetarium|power plant|reactor|substation|dam|water tower|cooling tower|oil rig|pumpjack|mine|quarry|sawmill|lumberyard|farm|ranch|orchard|vineyard|park|playground|fountain|plaza|square|courtyard|alley|street|road|highway|intersection|roundabout|tunnel|subway station|train station|airport|terminal|control tower|helipad|spaceport|launch pad|moonbase|habitat|biodome|dome|skyline|cityblock|skyscraper|highrise|office|apartment|condo|tenement|ruins|stonehenge|totem)\b",
    "humanoid": r"\b(man|woman|boy|girl|kid|child|baby|human|person|people|guy|lady|gentleman|villager|citizen|peasant|farmer|miner|blacksmith|merchant|trader|shopkeeper|baker|chef|cook|waiter|bartender|butcher|barber|tailor|carpenter|mason|builder|worker|mechanic|engineer|scientist|professor|teacher|student|doctor|nurse|surgeon|paramedic|firefighter|fireman|policeman|police|cop|detective|sheriff|deputy|guard|security|warden|prisoner|inmate|convict|criminal|thief|burglar|bandit|outlaw|robber|gangster|mafia|hitman|assassin|spy|agent|soldier|trooper|marine|commando|sniper|medic|pilot|captain|general|commander|admiral|sailor|seaman|pirate|buccaneer|corsair|viking|berserker|gladiator|spartan|centurion|legionnaire|knight|paladin|crusader|templar|squire|warrior|fighter|barbarian|brawler|monk|sensei|ninja|shinobi|samurai|ronin|shogun|kensei|archer|ranger|hunter|huntress|scout|rogue|bard|minstrel|jester|clown|mime|acrobat|magician|illusionist|wizard|sorcerer|sorceress|witch|warlock|mage|archmage|necromancer|druid|shaman|cleric|priest|priestess|bishop|pope|monarch|king|queen|prince|princess|emperor|empress|pharaoh|sultan|tsar|lord|duke|duchess|baron|noble|aristocrat|president|mayor|judge|lawyer|athlete|runner|sprinter|swimmer|diver|gymnast|boxer|wrestler|karate|judoka|fencer|skater|skier|snowboarder|surfer|cyclist|biker|racer|driver|golfer|tennis|basketball|football|soccer|baseball|hockey|quarterback|goalie|referee|coach|cheerleader|dancer|ballerina|singer|rockstar|musician|guitarist|drummer|dj|actor|director|artist|painter|sculptor|photographer|journalist|reporter|astronaut|cosmonaut|spaceman|explorer|adventurer|mountaineer|cowboy|cowgirl|desperado|native|tribal|nomad|bedouin|eskimo|inuit|elf|high elf|wood elf|dark elf|drow|dwarf|gnome|halfling|hobbit|orc|uruk|goblin|hobgoblin|troll|ogre|cyclops|giant|titan|goliath|minotaur|centaur|satyr|faun|harpy|siren|mermaid|merman|triton|naga|gorgon|medusa|lamia|succubus|incubus|demon|devil|fiend|imp|angel|seraph|cherub|valkyrie|deity|god|goddess|avatar|demigod|lich|vampire|nosferatu|werewolf|lycan|zombie|ghoul|wight|draugr|mummy|skeleton|ghost|phantom|specter|wraith|banshee|poltergeist|spirit|reaper|grim reaper|scarecrow|golem|ent|treant|elemental|djinn|genie|efreet|robot|android|cyborg|automaton|droid|mech|mecha|walker|synth|reploid|bionic|alien|martian|grey|reptilian|predator|mutant|superhero|supervillain|hero|villain|vigilante|sidekick|champion|character|npc|boss|miniboss|mob|dummy|mannequin|statue|bust|figurine|doll|action figure|toy soldier|nutcracker|snowman|santa|leprechaun|cupid)\b",
    "plant": r"\b(tree|oak|pine|fir|spruce|cedar|redwood|sequoia|birch|maple|willow|weeping willow|elm|ash|beech|poplar|aspen|cypress|yew|baobab|acacia|banyan|mangrove|palm|coconut palm|date palm|banana tree|apple tree|cherry blossom|sakura|peach tree|orange tree|lemon tree|olive tree|fig tree|bonsai|topiary|dead tree|hollow tree|stump|log|branch|twig|root|bush|shrub|hedge|bramble|briar|berry bush|rose bush|hydrangea|azalea|rhododendron|lavender|heather|sagebrush|tumbleweed|flower|rose|tulip|sunflower|daisy|lily|waterlily|lotus|orchid|iris|peony|poppy|marigold|dandelion|violet|pansy|carnations|chrysanthemum|daffodil|hyacinth|hibiscus|jasmine|plumeria|bougainvillea|bird of paradise|protea|edelweiss|blossom|bouquet|wreath|garland|petal|bud|stem|leaf|leaves|clover|shamrock|fern|frond|moss|lichen|ivy|vine|creeper|liana|grapevine|beanstalk|grass|lawn|turf|meadow|reed|bulrush|cattail|papyrus|bamboo|sugarcane|wheat|barley|oats|rye|corn|maize|rice|crop|cactus|saguaro|prickly pear|barrel cactus|agave|aloe|yucca|succulent|echeveria|jade plant|venus flytrap|pitcher plant|sundew|carnivorous plant|rafflesia|corpse flower|kelp|seaweed|algae|coral|anemone|sponge|mushroom|toadstool|fungi|fungus|morel|truffle|chanterelle|puffball|amanita|mycelium|spore|forest|woods|grove|thicket|jungle|rainforest|swamp|marsh|bog|oasis|garden|terrace|planter|terrarium)\b",
    "prop": r"\b(sword|broadsword|longsword|shortsword|greatsword|claymore|rapier|foil|saber|sabre|scimitar|cutlass|katana|wakizashi|nodachi|tanto|dagger|dirk|stiletto|knife|machete|kukri|cleaver|bayonet|blade|excalibur|axe|battleaxe|hatchet|tomahawk|halberd|poleaxe|glaive|bardiche|scythe|sickle|kama|spear|lance|pike|javelin|harpoon|trident|bident|pitchfork|staff|quarterstaff|bo staff|magic staff|crozier|scepter|wand|rod|club|mace|morningstar|flail|warhammer|maul|mallet|hammer|sledgehammer|pickaxe|mattock|crowbar|wrench|spanner|screwdriver|pliers|saw|chainsaw|drill|jackhammer|shovel|spade|hoe|rake|trowel|shears|scissors|anvil|vise|bellows|tongs|chisel|level|tape measure|toolbox|ladder|scaffold|wheelbarrow|bow|longbow|shortbow|recurve bow|compound bow|crossbow|ballista|catapult|trebuchet|cannon|mortar|howitzer|gatling|minigun|gun|handgun|pistol|revolver|deagle|glock|luger|flintlock|musket|blunderbuss|rifle|carbine|assault rifle|ak47|m16|sniper rifle|shotgun|blaster|raygun|laser gun|plasma rifle|railgun|gauss|phaser|needler|bazooka|rpg|missile launcher|grenade launcher|flamethrower|taser|grenade|bomb|dynamite|tnt|c4|landmine|claymore mine|torpedo|missile|nuke|ammo|bullet|shell|cartridge|quiver|arrow|bolt|dart|shuriken|kunai|boomerang|chakram|sling|slingshot|whip|lasso|net|bola|shield|buckler|heater shield|kite shield|tower shield|pavise|aegis|riot shield|armor|breastplate|cuirass|chainmail|plate mail|gauntlet|bracer|greave|pauldron|helmet|helm|great helm|sallet|bascinet|morion|kabuto|spartan helmet|viking helmet|knight helmet|space helmet|diving helmet|gas mask|mask|oni mask|tiki mask|venetian mask|goggles|visor|glasses|sunglasses|monocle|eyepatch|crown|tiara|diadem|circlet|coronet|halo|hat|cap|top hat|bowler|fedora|sombrero|cowboy hat|beret|beanie|fez|turban|straw hat|wizard hat|witch hat|pirate hat|tricorne|santa hat|chef hat|hard hat|ring|amulet|talisman|pendant|necklace|locket|choker|bracelet|bangle|brooch|earring|medal|medallion|badge|crest|emblem|belt|buckle|strap|holster|scabbard|sheath|cape|cloak|scarf|tie|bowtie|glove|mitten|boot|shoe|sneaker|sandal|slipper|skate|rollerblade|ski|snowshoe|flipper|backpack|rucksack|knapsack|satchel|messenger bag|briefcase|suitcase|luggage|trunk|duffel|purse|handbag|pouch|wallet|canteen|flask|waterskin|chest|treasure chest|coffer|strongbox|safe|lockbox|mimic|crate|box|carton|package|parcel|present|gift|giftbox|barrel|keg|cask|vat|tun|bucket|pail|tub|trough|basket|hamper|urn|vase|amphora|jug|pitcher|ewer|jar|mason jar|canister|tin|can|bottle|potion|elixir|vial|phial|beaker|test tube|retort|alembic|cauldron|kettle|teapot|coffee pot|mug|cup|goblet|chalice|grail|tankard|stein|wineglass|champagne flute|tumbler|shot glass|bowl|plate|platter|dish|tray|salver|saucer|tureen|gravy boat|fork|spoon|ladle|spatula|whisk|rolling pin|cutting board|mortar and pestle|salt shaker|pepper mill|napkin|tablecloth|placemat|coaster|candlestick|candelabra|menorah|chandelier|lantern|oil lamp|genie lamp|lava lamp|desk lamp|floor lamp|streetlamp|flashlight|torch|brazier|campfire|bonfire|fireplace|hearth|stove|oven|microwave|toaster|blender|mixer|coffee maker|espresso|fridge|refrigerator|freezer|sink|faucet|tap|bathtub|shower|toilet|mirror|scale|towel rack|soap dispenser|toothbrush|hairdryer|razor|comb|brush|perfume|lipstick|compact|table|dining table|coffee table|end table|side table|nightstand|desk|workbench|counter|island|altar|podium|lectern|easel|chair|armchair|recliner|rocking chair|office chair|gaming chair|stool|barstool|bench|pew|throne|sofa|couch|loveseat|sectional|futon|chaise|ottoman|pouf|beanbag|bed|bunk bed|four-poster|crib|cradle|hammock|mattress|pillow|cushion|blanket|quilt|rug|carpet|mat|curtain|drapes|blinds|screen|divider|wardrobe|armoire|closet|dresser|chest of drawers|cabinet|cupboard|hutches|pantry|sideboard|buffet|credenza|locker|bookshelf|bookcase|shelf|rack|coat rack|hat stand|umbrella stand|plant stand|pedestal|plinth|column|pillar|balustrade|railing|fence|mailbox|fire hydrant|trash can|dumpster|recycle bin|lamppost|traffic light|stop sign|street sign|billboard|neon sign|awnings|parasol|umbrella|sleeping bag|cooler|grill|bbq|picnic table|birdbath|birdhouse|feeder|gnomes|wind chime|weathervane|sundial|globe|orrery|telescope|microscope|binoculars|spyglass|magnifying glass|compass|sextant|astrolabe|hourglass|clock|grandfather clock|wall clock|alarm clock|pocket watch|wristwatch|stopwatch|metronome|thermometer|barometer|balance|weights|abacus|calculator|typewriter|cash register|atm|vending machine|arcade|arcade cabinet|pinball|claw machine|jukebox|slot machine|roulette|pool table|foosball|air hockey|dartboard|chessboard|chess piece|dice|domino|cards|poker chip|mahjong|puzzle|rubiks cube|yo-yo|spinning top|dreidel|marbles|blocks|lego|train set|rocking horse|teddy bear|plushie|kite|balloon|ball|soccer ball|basketball|football|baseball|tennis ball|golf ball|bowling ball|bowling pin|billiard ball|beach ball|snowglobe|trophy|ribbon|plaque|bell|gong|whistle|bugle|trumpet|trombone|tuba|french horn|saxophone|clarinet|oboe|flute|piccolo|recorder|panpipes|harmonica|accordion|bagpipes|organ|piano|grand piano|upright piano|keyboard|harpsichord|harp|lyre|lute|guitar|acoustic guitar|electric guitar|bass|ukulele|banjo|mandolin|sitar|violin|viola|cello|double bass|fiddle|drum|snare|kick drum|tom|bongo|conga|djembe|timpani|tambourine|maraca|castanets|triangle|xylophone|marimba|vibraphone|glockenspiel|cymbal|hi-hat|microphone|amp|amplifier|speaker|subwoofer|boombox|radio|gramophone|phonograph|turntable|record player|vinyl|cassette|walkman|mp3|ipod|headphones|earbuds|camera|dslr|polaroid|camcorder|video camera|projector|film reel|clapperboard|tripod|spotlight|tv|television|crt|flatscreen|monitor|computer|pc|laptop|macbook|tablet|ipad|phone|smartphone|iphone|rotary phone|payphone|telephone|fax|printer|scanner|router|modem|server rack|mainframe|hard drive|floppy|usb|sd card|cd|dvd|battery|solar panel|wind turbine|generator|transformer|engine|motor|piston|gear|cog|spring|lever|pulley|valve|pipe|gauge|meter|switch|button|control panel|console|radar|satellite dish|antenna|buoy|anchor|ship wheel|propeller|oar|paddle|lifebuoy|treasure map|scroll|parchment|book|tome|grimoire|spellbook|bible|dictionary|encyclopedia|notebook|journal|diary|ledger|clipboard|folder|binder|letter|envelope|stamp|newspaper|magazine|comic|poster|painting|frame|canvas|palette|paintbrush|pencil|pen|quill|inkwell|crayon|marker|chalk|eraser|ruler|protractor|stapler|tape|glue|paperclip|pushpin|sculpture|gargoyle|well|gravestone|tombstone|crypt|sarcophagus|coffin|casket|skull|bones|ribcage|fossil|amber|crystal|quartz|amethyst|emerald|ruby|sapphire|diamond|topaz|opal|pearl|geode|ore|ingot|gold bar|nugget|coin|doubloon|key|skeleton key|keycard|padlock|chain|shackles|handcuffs|cage|trap|bear trap|mousetrap|target|bullseye|sandcastle|cake|birthday cake|wedding cake|cupcake|muffin|donut|doughnut|croissant|bagel|bread|loaf|baguette|pretzel|toast|pancake|waffle|pie|tart|cookie|biscuit|macaron|brownie|chocolate|candy|lollipop|candy cane|gummy|ice cream|sundae|popsicle|milkshake|smoothie|pizza|burger|hamburger|cheeseburger|hotdog|sandwich|taco|burrito|nachos|sushi|nigiri|maki|ramen|noodles|soup|stew|salad|steak|rib|chicken leg|drumstick|ham|bacon|sausage|egg|fried egg|omelette|cheese|butter|milk|yogurt|honey|jam|peanut butter|ketchup|mustard|mayo|hot sauce|soy sauce|oil|vinegar|salt|pepper|sugar|flour|pasta|cereal|popcorn|chips|fries|nut|acorn|pinecone|apple|banana|orange|lemon|lime|grapefruit|peach|pear|plum|cherry|strawberry|raspberry|blueberry|blackberry|grape|watermelon|melon|cantaloupe|pineapple|mango|papaya|kiwi|pomegranate|fig|date|coconut|avocado|tomato|potato|carrot|broccoli|cauliflower|cabbage|lettuce|spinach|kale|celery|asparagus|artichoke|bell pepper|chili|jalapeno|onion|garlic|leek|radish|turnip|beet|pumpkin|squash|zucchini|cucumber|eggplant|aubergine|pea|bean|peanut)\b",
}
SUB_KW = {
    "dragon": r"\b(dragon|wyvern|drake|lindworm|wyrm|hydradragon|bahamut|smaug|charizard|balrog)\b",
    "bird": r"\b(bird|phoenix|eagle|hawk|falcon|owl|parrot|macaw|cockatoo|toucan|peacock|swan|flamingo|penguin|duck|goose|pelican|stork|crane|heron|ibis|vulture|condor|albatross|seagull|puffin|kiwi|ostrich|emu|cassowary|chicken|rooster|hen|turkey|quail|pheasant|pigeon|dove|crow|raven|magpie|jay|robin|sparrow|finch|canary|hummingbird|kingfisher|woodpecker|griffin|griffon|hippogriff|roc|harpy|thunderbird|cockatrice|archaeopteryx|pterodactyl|pteranodon|quetzalcoatlus)\b",
    "multi": r"\b(kraken|octopus|squid|cuttlefish|nautilus|jellyfish|medusa|hydra|cthulhu|cephalopod|spider|tarantula|black widow|recluse|scorpion|crab|hermit crab|king crab|lobster|crayfish|shrimp|prawn|mantis shrimp|isopod|trilobite|centipede|millipede|tick|mite|ant|termite|bee|bumblebee|wasp|hornet|yellowjacket|beetle|scarab|stag beetle|rhino beetle|ladybug|weevil|firefly|glowworm|cockroach|cricket|grasshopper|locust|katydid|mantis|praying mantis|stick insect|leaf insect|earwig|flea|louse|mosquito|fly|housefly|dragonfly|damselfly|butterfly|moth|caterpillar|silkworm|grub|maggot|larva|insect|bug|arachnid|crustacean|starfish|sea star|brittle star|sea urchin|sand dollar)\b",
    "aquatic": r"\b(fish|goldfish|koi|carp|trout|salmon|tuna|mackerel|sardine|anchovy|herring|cod|haddock|halibut|flounder|sole|plaice|turbot|catfish|pike|perch|bass|tilapia|piranha|barracuda|swordfish|marlin|sailfish|flying fish|anglerfish|lanternfish|blobfish|pufferfish|blowfish|boxfish|triggerfish|parrotfish|clownfish|angelfish|butterflyfish|tang|surgeonfish|moorish idol|lionfish|scorpionfish|stonefish|goby|blenny|wrasse|grouper|snapper|bream|betta|guppy|tetra|discus|cichlid|arowana|gar|arapaima|coelacanth|sturgeon|paddlefish|shark|great white|hammerhead|tiger shark|whale shark|basking shark|mako|thresher|bull shark|nurse shark|reef shark|goblin shark|megalodon|ray|stingray|manta ray|eagle ray|electric ray|sawfish|skate|guitarfish|whale|blue whale|humpback|sperm whale|orca|killer whale|beluga|narwhal|fin whale|minke|right whale|bowhead|pilot whale|beaked whale|dolphin|bottlenose|porpoise|vaquita|manatee|dugong|seacow|seal|sea lion|walrus|fur seal|elephant seal|leopard seal|otter|sea otter|platypus|axolotl|newt|salamander|tadpole|eel|moray|conger|electric eel|lamprey|hagfish|sea snake|serpent|sea serpent|leviathan|nessie|loch ness|mosasaur|ichthyosaur|plesiosaur|liopleurodon|basilosaurus|dunkleosteus|anomalocaris|seahorse|pipefish|seadragon|worm|earthworm|leech|bobbit worm|tube worm|flatworm|planarian|nudibranch|sea slug|sea cucumber|conch|snail|slug)\b",
}
WINGS_KW = r"\b(wing|wings|winged|dragon|wyvern|drake|pegasus|alicorn|griffin|griffon|hippogriff|phoenix|roc|thunderbird|harpy|gargoyle|sphinx|chimera|manticore|cockatrice|basilisk|bat|vampire bat|flying fox|pterodactyl|pteranodon|archaeopteryx|butterfly|moth|bee|bumblebee|wasp|hornet|dragonfly|damselfly|fairy|pixie|sprite|sylph|angel|seraph|cherub|valkyrie|demon|devil|imp|succubus|bird|eagle|hawk|falcon|owl|parrot|toucan|peacock|swan|flamingo|duck|goose|seagull|crow|raven|dove|pigeon|rooster|chicken|hummingbird|jetpack)\b"
AIR_SEA_KW = r"\b(plane|airplane|biplane|jet|fighter|helicopter|chopper|drone|blimp|zeppelin|rocket|spaceship|starship|spacecraft|shuttle|ufo|saucer|boat|ship|sailboat|yacht|speedboat|ferry|battleship|submarine|hovercraft|sled|sleigh|jetski|canoe|kayak|raft)\b"
BIG_VEHICLE_KW = r"\b(truck|pickup|lorry|semi|bus|van|suv|jeep|hummer|tractor|train|locomotive|tank|apc|rv|camper|forklift|bulldozer|excavator|crane|dump truck|cement mixer|combine|harvester|monster truck)\b"


def classify(prompt: str, hint: str = "") -> Tuple[str, str, bool, bool]:
    p, hint = prompt.lower(), (hint or "").lower().strip()
    matched = True
    if hint in PART_RANGE:
        cat = hint
    else:
        if any(re.search(SUB_KW[s], p) for s in ("dragon", "multi", "bird", "aquatic")):
            cat = "creature"
        else:
            cat = next((c for c in ("vehicle", "structure", "humanoid", "plant", "prop") if re.search(KW[c], p)), None)
            if cat is None:
                cat, matched = "creature", False
    sub = ""
    if cat == "creature":
        sub = next((s for s in ("dragon", "multi", "bird", "aquatic") if re.search(SUB_KW[s], p)), "quad")
    wings = bool(re.search(WINGS_KW, p)) and cat in ("creature", "humanoid") and sub != "dragon"
    return cat, sub, wings, matched


def part_range(cat: str, target: Optional[int]) -> Tuple[int, int]:
    lo, hi = PART_RANGE[cat]
    if target:
        t = max(30, min(165, int(target)))
        lo, hi = max(30, t - 12), min(165, t + 12)
    return lo, hi


def build_system_prompt(cat: str, sub: str, wings: bool, lo: int, hi: int) -> str:
    bp = BLUEPRINTS.get("creature_" + sub, BLUEPRINTS["creature_quad"]) if cat == "creature" else BLUEPRINTS[cat]
    blocks = [CORE_SKILL, bp] + ([WINGS_BP] if wings else [])
    return "\n\n".join(blocks).replace("@LO@", str(lo)).replace("@HI@", str(hi))


def build_user_prompt(prompt: str, cat: str, sub: str, lo: int, hi: int, web_ctx: str) -> str:
    out = [
        f'SUBJECT: "{prompt}"',
        f"CATEGORY: {cat}" + (f" ({sub})" if sub else ""),
        f"TARGET: {lo}-{hi} parts, longest side 24-45 studs.",
        "CRITICAL CLOUD 3D MODELING REQUIREMENTS:",
        "1) CLEAR 3D FACE RELIEF AT +Z: Place the face features in front of the head (`z = 7.5..9.8`): cream FacePlate, angled Brows, fierce red Eye_L & Eye_R, dark Pupils, protruding Upper Muzzle, Lower Jaw, dark Mouth_Cavity, pink Tongue, and 6 white Fangs/Teeth!",
        "2) SMOOTH MUSCULAR RIGGING: Use `smooth: 0.32..0.45` on all muscles, torso, head, and limbs. Align Shoulder -> Bicep -> Elbow -> Forearm -> Wrist Bracer -> Hand smoothly along the same X axis without jagged sideways steps.",
        "3) CLEAN COLOR BLOCKING: Dark navy-charcoal (`#18202c`) fur, warm cream (`#e8dcc4`) on face/chest/abs/fingers/toes, royal gold (`#f5c518`) on the head Crown and Wrist Bracers ONLY (no red on chest or hands!).",
    ]
    if web_ctx:
        out.append("LIVE WEB REFERENCE:\n" + web_ctx)
    return "\n".join(out)


def compact_build(parts: List[Dict[str, Any]]) -> str:
    rows = [[p["id"], p["shape"], [round(v, 2) for v in p["size"]], [round(v, 2) for v in p["pos"]],
             [round(v, 1) for v in p["rot"]], p["color"], round(p["smooth"], 2)] for p in parts]
    return json.dumps(rows, separators=(",", ":"))


def build_revision_prompt(prompt: str, cat: str, lo: int, hi: int, parts: List[Dict[str, Any]],
                          problems: List[str], change: str = "") -> str:
    out = [f'SUBJECT: "{prompt}"', f"CATEGORY: {cat}", f"TARGET: {lo}-{hi} parts.",
           'CURRENT BUILD in studs, rows are [id, shape, size, pos, rot, color, smooth]:', compact_build(parts)]
    if change:
        out.append(f'REQUESTED CHANGE FROM THE USER: "{change}"\nApply it faithfully and keep everything else.')
    if problems:
        out.append("PROBLEMS FOUND BY THE CHECKER - fix all of them:\n" + "\n".join("- " + m for m in problems))
    out.append("Ensure the face (brows, glowing eyes, open mouth cavity, white fangs, tongue) protrudes clearly at +Z in front of the head, rigging joints flow smoothly with `smooth: 0.35`, and colors are clean.")
    out.append("Return the COMPLETE corrected build in the same JSON schema (plan + every op), never a diff.")
    return "\n".join(out)


# --------------------------------------------------------------------------- providers & Argon / Gemini 3.8 support
class ProviderError(Exception):
    def __init__(self, provider: str, status: int, message: str):
        super().__init__(message)
        self.provider, self.status, self.message = provider, status, message

    def to_http(self) -> HTTPException:
        code = 401 if self.status in (401, 403) else 404 if self.status == 404 else 429 if self.status == 429 else 502
        return HTTPException(status_code=code, detail=f"{self.provider}: {self.message}"[:400])


def normalize_gemini_model(name: str) -> str:
    m = (name or DEFAULT_MODELS["gemini"]).strip().lower()
    if m.startswith("models/"):
        m = m.split("models/", 1)[1]
    if m in GEMINI_MODEL_ALIASES:
        return GEMINI_MODEL_ALIASES[m]
    if "argon" in m:
        return "gemini-4-argon"
    if not m.startswith("gemini-"):
        if "lite" in m:
            return "gemini-3.5-flash-lite"
        return "gemini-3.8-flash"
    return m


def _err_text(res: httpx.Response) -> str:
    try:
        j = res.json()
        e = j.get("error", j) if isinstance(j, dict) else j
        if isinstance(e, dict):
            return str(e.get("message") or e)[:260]
        return str(e)[:260]
    except Exception:
        return res.text[:260]


_LIMIT_HINT = re.compile(r"too large|too high|too many|exceed|at most|less than or equal|no more than|"
                         r"maximum (?:allowed|value|number|output|completion)|which is the maximum|is the maximum")


def _limit_from(low: str, current: int) -> int:
    if not _LIMIT_HINT.search(low):
        return 0
    nums = [int(n) for n in re.findall(r"(?<![\d.])\d{3,7}(?![\d.])", low)]
    below = [n for n in nums if 256 <= n < current]
    return max(below) if below else 0


def _heal_payload(payload: Dict[str, Any], body: str) -> bool:
    low, changed = body.lower(), False
    for k in ("temperature", "response_format", "reasoning_effort", "top_p"):
        if k in payload and k in low:
            payload.pop(k)
            changed = True
    if "response_format" in payload and re.search(r"json_validate_failed|failed_generation|json mode|json_object", low):
        payload.pop("response_format")
        changed = True
    swapped = False
    if "max_tokens" in payload and "max_completion_tokens" in low:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
        swapped = changed = True
    if not swapped:
        for k in ("max_tokens", "max_completion_tokens"):
            if k in payload and "token" in low and _LIMIT_HINT.search(low):
                lim = _limit_from(low, payload[k])
                new = lim or max(1024, payload[k] // 2)
                if new < payload[k]:
                    payload[k] = new
                    changed = True
    return changed


def _retry_delay(res: httpx.Response, attempt: int) -> float:
    try:
        ra = float(res.headers.get("retry-after", ""))
    except ValueError:
        ra = 0.0
    return min(5.0, ra if ra > 0 else 1.1 * attempt)


def _worth_retrying(res: httpx.Response) -> bool:
    if res.status_code not in RETRY_STATUS:
        return False
    low = res.text.lower()
    return not ("insufficient_quota" in low or "billing" in low)


async def _post_healing(client: httpx.AsyncClient, provider: str, url: str, headers: Dict[str, str],
                        payload: Dict[str, Any]) -> Dict[str, Any]:
    heals = retries = 0
    while True:
        res = await client.post(url, headers=headers, json=payload)
        if res.status_code == 200:
            try:
                return res.json()
            except ValueError:
                raise ProviderError(provider, 502, "the provider returned a non-JSON response")
        if res.status_code == 400 and heals < 6 and _heal_payload(payload, res.text):
            heals += 1
            continue
        if retries < 2 and _worth_retrying(res):
            retries += 1
            await asyncio.sleep(_retry_delay(res, retries))
            continue
        raise ProviderError(provider, res.status_code, _err_text(res))


async def call_gemini(client, key, model, system, user) -> Tuple[str, str]:
    requested = (model or DEFAULT_MODELS["gemini"]).strip()
    primary = normalize_gemini_model(requested)
    chain: List[str] = []
    for m in (
        primary,
        "gemini-4-argon",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
    ):
        if m and m not in chain:
            chain.append(m)

    budget = THINKING_BUDGETS.get(THINKING_LEVEL, 4096)
    if "medium" in requested.lower():
        budget = 4096
    elif "high" in requested.lower():
        budget = 8192

    last: Optional[ProviderError] = None
    for m in chain:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
        gc: Dict[str, Any] = {
            "responseMimeType": "application/json",
            "temperature": 0.34,
            "maxOutputTokens": 32768,
        }
        if budget > 0 and "lite" not in m:
            gc["thinkingConfig"] = {"thinkingBudget": budget}
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gc,
        }
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        res = None
        heals = retries = 0
        while True:
            res = await client.post(url, headers=headers, json=payload)
            if res.status_code == 400 and heals < 5:
                low = res.text.lower()
                if "thinking" in low and "thinkingConfig" in gc:
                    gc.pop("thinkingConfig")
                    heals += 1
                    continue
                if "mime" in low and "responseMimeType" in gc:
                    gc.pop("responseMimeType")
                    heals += 1
                    continue
                if "maxoutputtokens" in low or "output token" in low:
                    new = _limit_from(low, gc["maxOutputTokens"]) or 16384
                    if new < gc["maxOutputTokens"]:
                        gc["maxOutputTokens"] = new
                        heals += 1
                        continue
            if retries < 1 and _worth_retrying(res):
                retries += 1
                await asyncio.sleep(_retry_delay(res, retries))
                continue
            break
        if res.status_code == 200:
            data = res.json()
            cands = data.get("candidates") or [{}]
            parts = (cands[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            if text:
                return text, m
            last = ProviderError("gemini", 502, f"{m} returned no text (finish: {cands[0].get('finishReason')})")
            continue
        last = ProviderError("gemini", res.status_code, _err_text(res))
        if res.status_code in (401, 403) or (res.status_code == 400 and "api key" in res.text.lower()):
            raise last
    raise last or ProviderError("gemini", 502, "Gemini request failed")


def _is_reasoning(model: str) -> bool:
    return model.lower().startswith(("o1", "o3", "o4", "gpt-5", "gpt-6"))


async def call_openai_compat(client, provider, base, key, model, system, user) -> Tuple[str, str]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
    }
    if provider == "openai" and _is_reasoning(model):
        payload["max_completion_tokens"] = 32000
        payload["reasoning_effort"] = "medium"
    else:
        payload["temperature"] = 0.34
        payload["max_tokens"] = 16000 if provider == "openai" else 12000
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = await _post_healing(client, provider, base + "/chat/completions", headers, payload)
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    return msg.get("content") or "", model


async def call_openai(client, key, model, system, user):
    return await call_openai_compat(client, "openai", "https://api.openai.com/v1", key,
                                    model or DEFAULT_MODELS["openai"], system, user)


async def call_groq(client, key, model, system, user):
    return await call_openai_compat(client, "groq", "https://api.groq.com/openai/v1", key,
                                    model or DEFAULT_MODELS["groq"], system, user)


async def call_anthropic(client, key, model, system, user) -> Tuple[str, str]:
    model = model or DEFAULT_MODELS["anthropic"]
    payload = {"model": model, "max_tokens": 16000, "temperature": 0.34, "system": system,
               "messages": [{"role": "user", "content": user}]}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    data = await _post_healing(client, "anthropic", "https://api.anthropic.com/v1/messages", headers, payload)
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"), model


CALLERS = {"gemini": call_gemini, "openai": call_openai, "anthropic": call_anthropic, "groq": call_groq}


async def call_pollinations(client, system: str, user: str) -> str:
    res = await client.post("https://text.pollinations.ai/openai", json={
        "model": "openai", "temperature": 0.34, "jsonMode": True,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    if res.status_code != 200:
        raise ProviderError("free-fallback", res.status_code, _err_text(res))
    try:
        data = res.json()
    except ValueError:
        return res.text
    if isinstance(data, dict) and ("ops" in data or "parts" in data):
        return json.dumps(data)
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "") if isinstance(data, dict) else res.text


async def call_llm(system: str, user: str, provider: str = "", model: str = "", api_key: str = "") -> Tuple[str, Dict[str, str]]:
    prov, key, model = (provider or "").lower().strip(), (api_key or "").strip(), (model or "").strip()
    if not prov and any(k in model.lower() for k in ("gemini", "flash", "argon")):
        prov = "gemini"
    async with httpx.AsyncClient(timeout=httpx.Timeout(170.0, connect=15.0)) as client:
        if prov in CALLERS and (key or ENV_KEYS[prov]):
            try:
                text, used = await CALLERS[prov](client, key or ENV_KEYS[prov], model, system, user)
                return text, {"provider": prov, "model": used}
            except ProviderError as e:
                if e.status != 429:
                    raise
        errors: List[str] = []
        for name in ("gemini", "anthropic", "openai", "groq"):
            if ENV_KEYS[name]:
                try:
                    text, used = await CALLERS[name](client, ENV_KEYS[name], model if name == prov else "", system, user)
                    return text, {"provider": name, "model": used}
                except ProviderError as e:
                    errors.append(f"{e.provider}: {e.message}")
        try:
            return await call_pollinations(client, system, user), {"provider": "free-fallback", "model": "pollinations"}
        except Exception as e:
            errors.append(str(e))
    raise HTTPException(status_code=502, detail=" | ".join(errors)[:400] or "All AI providers failed.")


# --------------------------------------------------------------------------- web research
_RESEARCH: "OrderedDict[str, str]" = OrderedDict()


async def fetch_anatomical_research(query: str, override_key: str = "") -> str:
    key = (override_key or SERPER_API_KEY).strip()
    if not key:
        return ""
    ck = query.lower().strip()
    if ck in _RESEARCH:
        return _RESEARCH[ck]

    async def one(client: httpx.AsyncClient, q: str) -> List[str]:
        res = await client.post("https://google.serper.dev/search",
                                headers={"X-API-KEY": key, "Content-Type": "application/json"}, json={"q": q, "num": 5})
        if res.status_code != 200:
            return []
        d, bits = res.json(), []
        kg = d.get("knowledgeGraph") or {}
        if kg.get("description"):
            bits.append(str(kg["description"]))
        for k, v in list((kg.get("attributes") or {}).items())[:5]:
            bits.append(f"{k}: {v}")
        ab = d.get("answerBox") or {}
        if ab.get("snippet") or ab.get("answer"):
            bits.append(str(ab.get("snippet") or ab.get("answer")))
        bits += [o["snippet"] for o in (d.get("organic") or [])[:4] if o.get("snippet")]
        return bits

    qs = [f"{query} appearance colors chest face markings", f"{query} roblox boss model design", f"{query} character anatomy"]
    try:
        async with httpx.AsyncClient(timeout=8.5) as client:
            results = await asyncio.gather(*(one(client, q) for q in qs), return_exceptions=True)
    except Exception:
        return ""
    seen, bits = set(), []
    for r in results:
        if isinstance(r, list):
            for b in r:
                if b not in seen:
                    seen.add(b)
                    bits.append(b)
    text = " | ".join(bits)[:1400]
    if text:
        _RESEARCH[ck] = text
        while len(_RESEARCH) > 128:
            _RESEARCH.popitem(last=False)
    return text


def clean_web_context(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    m = re.search(r"Live Serper Web Research Context[^:\n]*:\s*(.+)$", raw, re.S | re.I)
    if m:
        return m.group(1).strip()[:1600]
    if raw.lower().startswith("request:"):
        return ""
    return raw[:1600]


# --------------------------------------------------------------------------- parsing & Cloud 3D Smoothing / Rigging / Facial Relief
def salvage_ops(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ops": []}
    m = re.search(r'"title"\s*:\s*"([^"]*)"', text)
    if m:
        out["title"] = m.group(1)
    m = re.search(r'"category"\s*:\s*"([^"]*)"', text)
    if m:
        out["category"] = m.group(1)
    m = re.search(r'"(?:ops|parts)"\s*:\s*\[', text)
    if not m:
        return out
    dec, i = json.JSONDecoder(), m.end()
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, k = dec.raw_decode(text, j)
        except ValueError:
            break
        if isinstance(obj, dict):
            out["ops"].append(obj)
        i = k
    return out


def parse_llm_json(text: str) -> Dict[str, Any]:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    t = re.sub(r"```(?:json)?", "", t, flags=re.I).strip()
    dec = json.JSONDecoder()
    for fix in (False, True):
        s = re.sub(r",\s*([}\]])", r"\1", t) if fix else t
        starts = [x for x in (s.find("{"), s.find("[")) if x >= 0]
        if not starts:
            break
        try:
            obj, _ = dec.raw_decode(s[min(starts):])
        except ValueError:
            continue
        if isinstance(obj, list):
            obj = {"ops": obj}
        if isinstance(obj, dict):
            return obj
    return salvage_ops(t)


def _num(v: Any, d: float = 0.0) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else d
    except (TypeError, ValueError):
        return d


def _vec3(v: Any, d: Tuple[float, float, float]) -> List[float]:
    if isinstance(v, (list, tuple)):
        return [_num(v[i], d[i]) if i < len(v) else d[i] for i in range(3)]
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return [float(v)] * 3
    return list(d)


def group_of(pid: str) -> str:
    return re.sub(r"\d+$", "", pid.split("_")[0])


def _row_to_op(row: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(row, (list, tuple)) or len(row) < 4 or not isinstance(row[0], str):
        return None
    out: Dict[str, Any] = {"id": row[0], "shape": row[1], "size": row[2], "pos": row[3]}
    rest = list(row[4:])
    if rest and isinstance(rest[0], (list, tuple)):
        out["rot"] = rest.pop(0)
    if rest and isinstance(rest[0], str):
        out["color"] = rest.pop(0)
    if rest and isinstance(rest[0], (int, float)) and not isinstance(rest[0], bool):
        out["smooth"] = rest.pop(0)
    return out


def _wrap_deg(a: float) -> float:
    return ((a + 180.0) % 360.0) - 180.0


def fix_cylinder_axes(pid: str, shape: str, size: List[float], rot: List[float]) -> Tuple[List[float], List[float]]:
    if shape not in ("cylinder", "cone"):
        return size, rot
    sx, sy, sz = size
    rx, ry, rz = rot
    is_wheel = bool(re.search(r"wheel|tire|tyre|rim|hub|axle|brake|caliper|coin|disc|disk|suck|pad|button", pid, re.I))

    if is_wheel and shape == "cylinder":
        dims = sorted([sx, sy, sz])
        thickness, diameter = dims[0], dims[2]
        if sx < sy * 0.8 and sx < sz * 0.8 and abs(rz) < 15:
            rz = 90.0
        elif sz < sx * 0.8 and sz < sy * 0.8 and abs(rx) < 15 and abs(rz) < 15:
            rx = 90.0
        elif abs(rx) < 15 and abs(rz) < 15 and re.search(r"wheel|tire|tyre|rim|hub|axle", pid, re.I):
            rz = 90.0
        return [diameter, thickness, diameter], [_wrap_deg(rx), _wrap_deg(ry), _wrap_deg(rz)]

    if sz > sy * 1.35 and abs(sx - sy) <= 0.42 * max(sx, sy):
        rad_avg = (sx + sy) / 2.0
        length = sz
        if abs(rx) < 45.0 and abs(rz) < 45.0:
            rx = _wrap_deg(rx + 90.0)
        return [rad_avg, length, rad_avg], [rx, ry, rz]

    if sx > sy * 1.35 and abs(sz - sy) <= 0.42 * max(sz, sy):
        rad_avg = (sy + sz) / 2.0
        length = sx
        if abs(rz) < 45.0 and abs(rx) < 45.0:
            rz = _wrap_deg(rz - 90.0)
        return [rad_avg, length, rad_avg], [rx, ry, rz]

    if max(sx, sz) > min(sx, sz) * 1.35:
        rad_avg = (sx + sz) / 2.0
        return [rad_avg, sy, rad_avg], [rx, ry, rz]

    return size, rot


_ROUND_EXEMPT = re.compile(
    r"eye|pupil|iris|orb|ball|pearl|gem|jewel|bead|knob|bubble|halo|lens|egg|berry|fruit|planet|globe|"
    r"joint|shoulder|elbow|knee|hip|ankle|wrist|dome|cranium|cheek|snout|muzzle|nose|chin|pec|bicep|thigh", re.I
)
_SPIKE_OK = re.compile(r"spike|thorn|fang|tooth|talon|claw|horn|tusk|stinger|crown_point|crown_spike", re.I)
_RING_OK = re.compile(r"ring|halo|hoop|tire|tyre|bracer|cuff|collar|crown_band", re.I)
_CYL_EXEMPT = re.compile(
    r"wheel|tire|tyre|rim|hub|axle|barrel|pipe|exhaust|column|pillar|trunk|wand|staff|handle|pole|post|"
    r"nozzle|thruster|cannon|muzzle|suck|candle|lamp|chimney|spar|grip|hilt|bracer|cuff|crown|bicep|forearm|thigh|shin|neck", re.I
)
_CRISP_SHARP = re.compile(
    r"tooth|teeth|fang|claw|talon|spike|thorn|blade|edge|sword|dagger|axe|crown_spike|crown_point|wall|floor|plinth|step|brick|crate", re.I
)
_ORGANIC_MUSCLE = re.compile(
    r"head|cranium|skull|dome|cheek|face|muzzle|snout|jaw|chin|brow|ear|tongue|torso|chest|pec|belly|ab|abs|pelvis|rump|hump|"
    r"shoulder|bicep|elbow|forearm|wrist|hand|palm|finger|thumb|hip|thigh|knee|shin|calf|ankle|foot|toe|"
    r"mantle|tentacle|arm|leg|tail|neck|fin|pauldron|tuft|mane|foliage|canopy|cap|mushroom", re.I
)


def heal_part_shape_and_smoothness(pid: str, shape: str, size: List[float], smooth: float, cat: str, sub: str) -> Tuple[str, float]:
    """Cloud 3D Organic Smoothing Policy:
    Allows spherical joints & rounded organic bevels on creatures/humanoids/plants so models
    look smoothly sculpted instead of sketchy raw blocks."""
    longest = max(size)
    if shape == "sphere":
        if not (_ROUND_EXEMPT.search(pid) or (cat in ("creature", "humanoid", "plant") and longest <= 8.5)):
            shape = "box"
    elif shape == "cone":
        if not (_SPIKE_OK.search(pid) and longest <= 4.0):
            shape = "box"
    elif shape == "torus":
        if not _RING_OK.search(pid):
            shape = "box"

    # Compute adaptive organic bevel smoothness
    if _CRISP_SHARP.search(pid):
        smooth = min(smooth, 0.06)
    elif cat in ("creature", "humanoid", "plant") and _ORGANIC_MUSCLE.search(pid):
        if shape == "box":
            smooth = max(smooth, 0.34)
        elif shape == "cylinder":
            smooth = max(smooth, 0.40)
        elif shape == "sphere":
            smooth = max(smooth, 0.68)
    elif shape == "box":
        smooth = max(0.08, min(smooth, 0.32))
    return shape, max(0.0, min(1.0, smooth))


def enforce_roblox_budget(parts: List[Dict[str, Any]], cat: str) -> int:
    n = len(parts)
    if n < 8:
        return 0
    changed = 0
    round_idx = sorted((i for i, p in enumerate(parts) if p["shape"] in ("sphere", "cone", "torus")),
                       key=lambda i: (bool(_ROUND_EXEMPT.search(parts[i]["id"])), -vol(parts[i])))
    # Allow up to 22% spheres/cones for smooth rigging joints, domes, eyes, and fangs
    cap_round = max(6, int(0.22 * n))
    for i in round_idx[: max(0, len(round_idx) - cap_round)]:
        parts[i]["shape"] = "box"
        parts[i]["smooth"] = max(parts[i]["smooth"], 0.34)
        changed += 1
    if cat in ("creature", "humanoid", "plant"):
        cyl = sorted((i for i, p in enumerate(parts) if p["shape"] == "cylinder" and not _CYL_EXEMPT.search(p["id"])),
                     key=lambda i: -vol(parts[i]))
        cap_cyl = max(4, int(0.20 * n))
        for i in cyl[cap_cyl:]:
            parts[i]["shape"] = "box"
            parts[i]["smooth"] = max(parts[i]["smooth"], 0.34)
            changed += 1
    return changed


def _dist90(a: float) -> float:
    return abs(((a + 45.0) % 90.0) - 45.0)


def lean_deg(p: Dict[str, Any]) -> float:
    return max(_dist90(p["rot"][0]), _dist90(p["rot"][2]))


_LEAN_OK = re.compile(r"tail|wing|spar|feather|roof|hood|windshield|glass|slope|ramp|horn|tusk|branch|blade|antler|thorn|spike|stair|gable|ridge|brow|pauldron|tuft|crown", re.I)
_LIMB_RE = re.compile(r"arm|tentacle|leg|neck|finger|toe|claw|seg|bicep|forearm|thigh|shin|jaw|bracer|cuff", re.I)


def limit_lean(parts: List[Dict[str, Any]], max_share: float = 0.38) -> int:
    n = len(parts)
    if n < 20:
        return 0
    fixed = 0

    def flatten(p: Dict[str, Any]) -> None:
        p["rot"][0] = float(round(p["rot"][0] / 90.0) * 90)
        p["rot"][2] = float(round(p["rot"][2] / 90.0) * 90)

    for p in parts:
        if p["shape"] == "box" and lean_deg(p) > 2 and not _LEAN_OK.search(p["id"]) and not _LIMB_RE.search(p["id"]):
            flatten(p)
            fixed += 1
    leaning = sorted((p for p in parts if p["shape"] == "box" and lean_deg(p) > 2 and not _LEAN_OK.search(p["id"])),
                     key=lean_deg)
    over = sum(1 for p in parts if lean_deg(p) > 2) - int(max_share * n)
    for p in leaning:
        if over <= 0:
            break
        flatten(p)
        fixed += 1
        over -= 1
    return fixed


# --------------------------------------------------------------------------- Word-Boundary Semantic Color-Zone Enforcer (Fixed in v18)
def _hex_to_hls(c: str) -> Tuple[float, float, float]:
    r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h: float, l: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, max(0.0, min(1.0, l)), max(0.0, min(1.0, s)))
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def _color_dist(c1: str, c2: str) -> float:
    r1, g1, b1 = (int(c1[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    r2, g2, b2 = (int(c2[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    return math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2)


# Token-boundary regexes so 'Torso_Core' or 'Hand_Core' NEVER match 'core' or 'iris'!
_TOK = r"(?:^|_)"
_TOK_END = r"(?:$|_|\d)"

_ZONE_SKIN_CHEST = re.compile(
    _TOK + r"(?:chest|pec|pecs|ab|abs|belly|underbelly|stomach|scute|face|faceplate|faceupper|cheekbone|"
           r"muzzle|snout|jaw|brow|ear|palm|finger|thumb|toe|suck|throat)" + _TOK_END, re.I
)
_ZONE_GOLD_TRIM = re.compile(
    _TOK + r"(?:crown|bracer|cuff|bangle|buckle|trim|emblem|collar|rim|hilt|guard|pommel|halo|tiara)" + _TOK_END, re.I
)
_ZONE_GLOW_ACCENT = re.compile(
    _TOK + r"(?:eye|eyes|iris|visor|reactor|gem|jewel|rune|flame|fire|laser|plasma|taillight|thruster|afterburner)" + _TOK_END, re.I
)
_ZONE_WHITE_LIGHT = re.compile(
    _TOK + r"(?:tooth|teeth|fang|fangs|tusk|claw|talon|nail|sclera|highlight|shine|glint|headlight)" + _TOK_END, re.I
)
_ZONE_DARK = re.compile(
    _TOK + r"(?:pupil|mouth|cavity|nostril|tire|tyre|slit|grille)" + _TOK_END, re.I
)
_ZONE_TONGUE = re.compile(_TOK + r"(?:tongue|gum|gums)" + _TOK_END, re.I)
_ZONE_FORCE_BASE = re.compile(
    _TOK + r"(?:cranium|topdome|cheek|chin|torso|pelvis|hump|back|pauldron|tuft|shoulder|bicep|elbow|forearm|hand|hip|thigh|knee|shin|calf|foot|ankle)" + _TOK_END, re.I
)


def enforce_color_blocking(parts: List[Dict[str, Any]], palette: Dict[str, Any], prompt: str, cat: str) -> None:
    p_low = prompt.lower()
    is_ape = any(w in p_low for w in ("gorilla", "ape", "monkey", "kong", "primate", "baboon", "orangutan", "chimpanzee"))
    base = str(palette.get("base") or "#18202c").lower()
    if not re.fullmatch(r"#[0-9a-f]{6}", base):
        base = "#18202c"

    bh, bl, bs = _hex_to_hls(base)
    if is_ape or (cat == "creature" and (bs < 0.12 or bl > 0.35)):
        base = "#18202c"  # Deep navy-charcoal fur matching Roblox Gorilla Boss
        bh, bl, bs = _hex_to_hls(base)

    sec = str(palette.get("secondary") or "").lower()
    if not re.fullmatch(r"#[0-9a-f]{6}", sec) or _color_dist(base, sec) < 0.32 or _hex_to_hls(sec)[1] < 0.60:
        if is_ape or any(w in p_low for w in ("bear", "wolf", "lion", "dog", "cat", "rex", "dino", "dragon")):
            sec = "#e8dcc4"  # Warm sculpted cream skin/chest/faceplate
        else:
            sec = _hls_to_hex((bh + 0.08) % 1.0, 0.82, 0.35)

    trim = str(palette.get("trim") or "#f5c518").lower()
    if not re.fullmatch(r"#[0-9a-f]{6}", trim) or _color_dist(base, trim) < 0.28:
        trim = "#f5c518"  # Royal Gold for Crown & Wrist Bracers

    accent = str(palette.get("accent") or "#ff2a2a").lower()
    if not re.fullmatch(r"#[0-9a-f]{6}", accent) or _color_dist(base, accent) < 0.25:
        accent = "#ff2a2a"

    light = "#ffffff"
    dark = "#0d1017"

    palette["base"] = base
    palette["secondary"] = sec
    palette["trim"] = trim
    palette["accent"] = accent
    palette["light"] = light
    palette["dark"] = dark

    for p in parts:
        pid = p["id"]
        if _ZONE_TONGUE.search(pid):
            p["color"] = "#e06b80"
        elif _ZONE_WHITE_LIGHT.search(pid):
            p["color"] = light
        elif _ZONE_DARK.search(pid):
            p["color"] = dark
        elif _ZONE_GLOW_ACCENT.search(pid) and not re.search(r"brow|socket|lid|frame|crown", pid, re.I):
            p["color"] = accent
        elif _ZONE_GOLD_TRIM.search(pid) and not re.search(r"innerdome|waist|belt|pelvis", pid, re.I):
            p["color"] = trim
        elif _ZONE_SKIN_CHEST.search(pid) and not re.search(r"cranium|cheek_l|cheek_r|chin_fur|back|shoulder|pauldron|bicep|forearm|thigh|shin|hand_l|hand_r|foot_l|foot_r", pid, re.I):
            p["color"] = sec
        elif is_ape and _ZONE_FORCE_BASE.search(pid):
            if re.search(r"pauldron|tuft|back_silver|saddle", pid, re.I):
                p["color"] = _hls_to_hex(bh, min(0.26, bl + 0.06), max(0.22, bs))
            else:
                p["color"] = base


# --------------------------------------------------------------------------- Cloud 3D Facial Depth Relief & Rigging Welder (NEW in v18)
def extrude_facial_hierarchy(parts: List[Dict[str, Any]], cat: str) -> int:
    """Guarantees the creature's face is NEVER swallowed by the head block!
    Finds the head cranium/core front Z plane and steps facial parts forward in 4 clear relief layers:
    Layer 1 (Cranium front) -> Layer 2 (FacePlate / Muzzle / Jaw) -> Layer 3 (Brows / Eyes / Mouth Cavity)
    -> Layer 4 (Pupils / Fangs / Teeth / Tongue / Nostrils)."""
    if cat not in ("creature", "humanoid"):
        return 0

    head_cores = [
        p for p in parts
        if re.search(r"^(?:head|skull|cranium)", p["id"], re.I)
        and not re.search(r"face|muzzle|snout|jaw|mouth|cavity|tongue|tooth|teeth|fang|eye|pupil|brow|nostril|ear|horn|crown|crest", p["id"], re.I)
    ]
    if not head_cores:
        return 0

    # Find front-most Z surface of the main cranium
    cranium_front_z = max(p["pos"][2] + p["size"][2] * 0.5 for p in head_cores)
    adjusted = 0

    # Layer 2: FacePlate, Upper Snout/Muzzle, Lower Jaw -> sits on or protrudes in front of cranium_front_z
    face_front_z = cranium_front_z + 0.55
    for p in parts:
        pid = p["id"]
        if re.search(r"faceplate|faceupper|face_plate|cheekbone", pid, re.I):
            half_z = p["size"][2] * 0.5
            target_front = cranium_front_z + 0.65
            if p["pos"][2] + half_z < target_front:
                p["pos"][2] = target_front - half_z
                adjusted += 1
            face_front_z = max(face_front_z, p["pos"][2] + half_z)

    muzzle_front_z = face_front_z + 0.75
    for p in parts:
        pid = p["id"]
        if re.search(r"muzzle|snout|jaw_lower|head_jaw", pid, re.I) and not re.search(r"cavity|mouth", pid, re.I):
            half_z = p["size"][2] * 0.5
            target_front = face_front_z + 0.85
            if p["pos"][2] + half_z < target_front:
                p["pos"][2] = target_front - half_z
                adjusted += 1
            muzzle_front_z = max(muzzle_front_z, p["pos"][2] + half_z)

    # Layer 3: Brow ridges, Glowing Eyes, Open Mouth Cavity
    for p in parts:
        pid = p["id"]
        half_z = p["size"][2] * 0.5
        if re.search(r"brow", pid, re.I):
            target_front = max(face_front_z + 0.75, muzzle_front_z + 0.15)
            if p["pos"][2] + half_z < target_front:
                p["pos"][2] = target_front - half_z
                adjusted += 1
        elif re.search(r"(?:^|_)eye(?:_|$|\d)", pid, re.I) and not re.search(r"pupil|brow|socket", pid, re.I):
            target_front = face_front_z + 0.48
            if p["pos"][2] + half_z < target_front:
                p["pos"][2] = target_front - half_z
                adjusted += 1
            p["size"][0] = max(p["size"][0], 1.65)
            p["size"][1] = max(p["size"][1], 0.90)
        elif re.search(r"mouth_cavity|mouthcavity|head_mouth", pid, re.I):
            target_front = muzzle_front_z - 0.10
            if p["pos"][2] + half_z < target_front:
                p["pos"][2] = target_front - half_z
                adjusted += 1

    # Layer 4: Pupils, Fangs/Teeth, Tongue, Nostrils (must sit proud of eyes & mouth cavity!)
    for p in parts:
        pid = p["id"]
        half_z = p["size"][2] * 0.5
        if re.search(r"pupil", pid, re.I):
            target_front = face_front_z + 0.72
            if p["pos"][2] + half_z < target_front:
                p["pos"][2] = target_front - half_z
                adjusted += 1
        elif re.search(r"fang|tooth|teeth", pid, re.I):
            target_front = muzzle_front_z + 0.22
            if p["pos"][2] + half_z < target_front:
                p["pos"][2] = target_front - half_z
                adjusted += 1
        elif re.search(r"tongue", pid, re.I):
            target_front = muzzle_front_z + 0.08
            if p["pos"][2] + half_z < target_front:
                p["pos"][2] = target_front - half_z
                adjusted += 1
        elif re.search(r"nostril", pid, re.I):
            target_front = muzzle_front_z + 0.25
            if p["pos"][2] + half_z < target_front:
                p["pos"][2] = target_front - half_z
                adjusted += 1

    return adjusted


def smooth_and_weld_rigging(parts: List[Dict[str, Any]], cat: str) -> int:
    """Fixes sketchy staircase rigging on arms and legs by aligning lateral X drift across limb chains
    and injecting rounded joint spheres where segments meet with a visible step."""
    if cat not in ("creature", "humanoid"):
        return 0

    welded = 0
    for side_sign, side_tag in ((-1.0, "l"), (1.0, "r")):
        arm_chain = []
        for stage_re, order in (
            (rf"arm_{side_tag}_shoulder|shoulder_{side_tag}", 1),
            (rf"arm_{side_tag}_bicep|arm_{side_tag}_upper|bicep_{side_tag}|upperarm_{side_tag}", 2),
            (rf"arm_{side_tag}_elbow|elbow_{side_tag}", 3),
            (rf"arm_{side_tag}_forearm|arm_{side_tag}_lower|forearm_{side_tag}", 4),
            (rf"bracer_{side_tag}|wrist_{side_tag}", 5),
            (rf"hand_{side_tag}", 6),
        ):
            matches = [p for p in parts if re.search(stage_re, p["id"], re.I) and not re.search(r"finger|thumb|pauldron|tuft", p["id"], re.I)]
            if matches:
                arm_chain.append((order, matches[0]))

        if len(arm_chain) >= 3:
            # Smooth lateral X alignment so the arm doesn't staircase outward/inward jaggedly
            xs = [p["pos"][0] for _, p in arm_chain]
            median_x = sum(xs) / len(xs)
            for order, p in arm_chain:
                target_x = median_x * (0.94 if order == 1 else 1.0)
                if abs(p["pos"][0] - target_x) > 0.8:
                    p["pos"][0] = 0.35 * p["pos"][0] + 0.65 * target_x
                    welded += 1
                p["smooth"] = max(p["smooth"], 0.38)

            # Also align all bracer pieces to forearm X/Z
            forearms = [p for ord_k, p in arm_chain if ord_k == 4]
            if forearms:
                fx, fz = forearms[0]["pos"][0], forearms[0]["pos"][2]
                for p in parts:
                    if re.search(rf"bracer_{side_tag}|cuff_{side_tag}", p["id"], re.I):
                        p["pos"][0] = fx
                        p["pos"][2] = fz
                        p["smooth"] = max(p["smooth"], 0.28)

    return welded


def _snap_q(v: float, q: float = 0.25) -> float:
    return round(v / q) * q


def _snap_rot(a: float) -> float:
    return _wrap_deg(round(a / 15.0) * 15.0)


def normalize_ops(raw_ops: List[Any], unit: float = 1.0, default_color: str = "#18202c",
                  cat: str = "creature", sub: str = "") -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(raw_ops):
        if len(parts) >= MAX_PARTS:
            break
        if isinstance(item, (list, tuple)):
            item = _row_to_op(item)
        if not isinstance(item, dict) or str(item.get("op", "add")).lower() not in ("add", "", "part"):
            continue
        shape = str(item.get("shape") or item.get("s") or item.get("type") or "box").lower().strip()
        shape = SHAPE_ALIAS.get(shape, shape)
        if shape not in SHAPES:
            shape = "box"
        pid = re.sub(r"[^A-Za-z0-9_]", "", str(item.get("id") or item.get("name") or item.get("n") or f"Part_{idx + 1}").replace(" ", "_"))[:36] or f"Part_{idx + 1}"
        base, n = pid, 2
        while pid in seen:
            pid, n = f"{base}_{n}", n + 1
        seen.add(pid)

        size = [min(MAX_PART_STUDS, max(MIN_PART_STUDS, abs(s) * unit))
                for s in _vec3(item.get("size") or item.get("sz") or item.get("scale"), (1, 1, 1))]
        rp = item.get("pos") or item.get("position") or item.get("po")
        if isinstance(rp, (list, tuple)):
            pos = [max(-90.0, min(90.0, v * unit)) for v in _vec3(rp, (0, 0, 0))]
        else:
            pos = [0.0, size[1] / 2, 0.0]
        rot = [_snap_rot(r) for r in _vec3(item.get("rot") or item.get("rotation") or item.get("ro"), (0, 0, 0))]

        smooth = max(0.0, min(1.0, _num(item.get("smooth"), SMOOTH_DEFAULT[shape])))
        shape, smooth = heal_part_shape_and_smoothness(pid, shape, size, smooth, cat, sub)
        size, rot = fix_cylinder_axes(pid, shape, size, rot)
        if shape == "box" and all(abs(((r + 45.0) % 90.0) - 45.0) < 0.6 for r in rot):
            size = [max(0.20, _snap_q(s, 0.1)) for s in size]
            pos = [_snap_q(v, 0.1) for v in pos]

        color = str(item.get("color") or item.get("c") or default_color).strip()
        if re.fullmatch(r"#[0-9a-fA-F]{3}", color):
            color = "#" + "".join(ch * 2 for ch in color[1:])
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            color = default_color
        parts.append({"id": pid, "shape": shape, "size": size, "pos": pos, "rot": rot, "color": color.lower(), "smooth": smooth})
    return parts


# --------------------------------------------------------------------------- geometry math & kinematic chain stitcher
def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rot_matrix(rot: List[float]):
    rx, ry, rz = (math.radians(v) for v in rot)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return _mat_mul(_mat_mul([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]),
                    [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])


def aabb(p: Dict[str, Any]) -> List[Tuple[float, float]]:
    w, h, z = p["size"]
    half_local = (0.47 * w, 0.47 * h, 0.15 * z) if p["shape"] == "torus" else (w / 2, h / 2, z / 2)
    r = rot_matrix(p["rot"])
    half = [sum(abs(r[i][j]) * half_local[j] for j in range(3)) for i in range(3)]
    return [(p["pos"][i] - half[i], p["pos"][i] + half[i]) for i in range(3)]


def aabb_gap(b1: List[Tuple[float, float]], b2: List[Tuple[float, float]]) -> Tuple[float, List[float]]:
    gaps = [max(b1[k][0] - b2[k][1], b2[k][0] - b1[k][1], 0.0) for k in range(3)]
    mv = [(b1[k][0] - b2[k][1]) if b1[k][0] > b2[k][1]
          else (b1[k][1] - b2[k][0]) if b2[k][0] > b1[k][1] else 0.0 for k in range(3)]
    return math.sqrt(sum(g * g for g in gaps)), mv


def vol(p: Dict[str, Any]) -> float:
    return p["size"][0] * p["size"][1] * p["size"][2]


def axis_aligned(p: Dict[str, Any]) -> bool:
    return all(abs(((r + 45.0) % 90.0) - 45.0) < 0.6 for r in p["rot"])


def bounds(parts: List[Dict[str, Any]]):
    bx = [aabb(p) for p in parts]
    lo = [min(b[k][0] for b in bx) for k in range(3)]
    hi = [max(b[k][1] for b in bx) for k in range(3)]
    return lo, hi


def frame(parts: List[Dict[str, Any]]) -> float:
    lo, hi = bounds(parts)
    m = max(hi[k] - lo[k] for k in range(3))
    s = MAX_EXTENT_STUDS / m if m > MAX_EXTENT_STUDS else 1.0
    cx, cz, y0 = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2, lo[1]
    for p in parts:
        p["pos"] = [(p["pos"][0] - cx) * s, (p["pos"][1] - y0) * s, (p["pos"][2] - cz) * s]
        if s != 1.0:
            p["size"] = [max(MIN_PART_STUDS, v * s) for v in p["size"]]
    return s


def dedupe(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for p in parts:
        k = (p["shape"], tuple(round(v, 1) for v in p["pos"]), tuple(round(v, 1) for v in p["size"]), tuple(round(v) for v in p["rot"]))
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


_CHAIN_RE = re.compile(
    r"^((?:tentacle|arm|leg|tail|neck|spine|wing|branch|root|horn|antenna|claw|finger|toe|vine|feeler|whisker|fin)"
    r"(?:_[a-zA-Z0-9]+?)?)"
    r"(?:_?(?:seg|part|sec|bone|joint|node|ring|lobe)?_?(\d+)|_(base|root|shoulder|bicep|upper|elbow|mid|middle|lower|forearm|thigh|knee|shin|calf|ankle|club|tip|end|paw|hand|foot|claw))$",
    re.I,
)
_STAGE_ORDER = {
    "base": 1, "root": 1, "shoulder": 1, "thigh": 1,
    "upper": 2, "bicep": 2, "elbow": 3, "knee": 3, "mid": 3, "middle": 3,
    "lower": 4, "forearm": 4, "shin": 4, "calf": 4, "ankle": 5,
    "club": 8, "paw": 8, "hand": 8, "foot": 8, "claw": 9, "tip": 10, "end": 10,
}
_DETAIL_CHILD_RE = re.compile(r"suck|pad|claw|nail|talon|spike|barb|tuft|spot|glow|ring|stripe|bracer|cuff", re.I)


def stitch_articulated_chains(parts: List[Dict[str, Any]]) -> int:
    chains: Dict[str, List[Tuple[int, int]]] = {}
    for idx, p in enumerate(parts):
        pid = p["id"]
        if _DETAIL_CHILD_RE.search(pid) and not re.search(r"club$", pid, re.I):
            continue
        m = _CHAIN_RE.match(pid)
        if not m:
            continue
        prefix = m.group(1).lower()
        order = int(m.group(2)) if m.group(2) else _STAGE_ORDER.get((m.group(3) or "").lower(), 5)
        chains.setdefault(prefix, []).append((order, idx))

    stitched = 0
    for prefix, seq in chains.items():
        if len(seq) < 2:
            continue
        seq.sort(key=lambda x: x[0])
        indices = [idx for _, idx in seq]
        chain_set = set(indices)
        child_indices = [
            i for i, p in enumerate(parts)
            if i not in chain_set and p["id"].lower().startswith(prefix)
        ]

        for pos_idx in range(len(indices) - 1):
            i_prev = indices[pos_idx]
            i_curr = indices[pos_idx + 1]
            p_prev, p_curr = parts[i_prev], parts[i_curr]
            b_prev, b_curr = aabb(p_prev), aabb(p_curr)
            gap, mv = aabb_gap(b_prev, b_curr)

            dx = [p_curr["pos"][k] - p_prev["pos"][k] for k in range(3)]
            dist = math.sqrt(sum(d * d for d in dx))
            r_prev = max(p_prev["size"]) * 0.5
            r_curr = max(p_curr["size"]) * 0.5
            max_reach = (r_prev + r_curr) * 0.88

            if gap > 0.02 or dist > max_reach * 1.15:
                long_axis = max(range(3), key=lambda k: p_curr["size"][k])
                if p_curr["shape"] in ("cylinder", "cone"):
                    long_axis = 1
                p_curr["size"][long_axis] = min(MAX_PART_STUDS, p_curr["size"][long_axis] * 1.18)
                b_curr = aabb(p_curr)
                gap, mv = aabb_gap(b_prev, b_curr)

            if gap > 0.01 and dist > 1e-4:
                shift = [v + (0.14 if v > 0 else -0.14 if v < 0 else 0.0) for v in mv]
                downstream = indices[pos_idx + 1:]
                curr_pos_before = list(p_curr["pos"])
                for d_idx in downstream:
                    for k in range(3):
                        parts[d_idx]["pos"][k] += shift[k]
                for c_idx in child_indices:
                    c_pos = parts[c_idx]["pos"]
                    d_to_curr = math.sqrt(sum((c_pos[k] - curr_pos_before[k]) ** 2 for k in range(3)))
                    d_to_prev = math.sqrt(sum((c_pos[k] - p_prev["pos"][k]) ** 2 for k in range(3)))
                    if d_to_curr <= d_to_prev * 1.25:
                        for k in range(3):
                            parts[c_idx]["pos"][k] += shift[k]
                stitched += 1
    return stitched


# --------------------------------------------------------------------------- checks and auto-fixes
def find_coplanar(parts: List[Dict[str, Any]]) -> List[Tuple[int, int, int, int]]:
    boxes = [(i, aabb(p)) for i, p in enumerate(parts) if p["shape"] == "box" and axis_aligned(p)]
    hits = []
    for a in range(len(boxes)):
        i, bi = boxes[a]
        for b in range(a + 1, len(boxes)):
            j, bj = boxes[b]
            if min(min(bi[k][1], bj[k][1]) - max(bi[k][0], bj[k][0]) for k in range(3)) <= 0.05:
                continue
            for k in range(3):
                if abs(bi[k][1] - bj[k][1]) < 0.025:
                    hits.append((i, j, k, 1))
                if abs(bi[k][0] - bj[k][0]) < 0.025:
                    hits.append((i, j, k, 0))
    return hits


def fix_coplanar(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    pairs, names = set(), []
    for pass_idx in range(8):
        hits = find_coplanar(parts)
        if not hits:
            break
        done = set()
        step_push = PUSH * (1.0 + 0.18 * pass_idx)
        for i, j, k, side in hits:
            if (i, j) not in pairs:
                pairs.add((i, j))
                names.append(f"{parts[i]['id']}/{parts[j]['id']}")
            tgt = i if vol(parts[i]) < vol(parts[j]) else j
            if (tgt, k, side) in done:
                continue
            done.add((tgt, k, side))
            parts[tgt]["size"][k] += step_push
            parts[tgt]["pos"][k] += step_push / 2 if side == 1 else -step_push / 2
    return {"pairs": len(pairs), "names": names, "remaining": len({(i, j) for i, j, _, _ in find_coplanar(parts)})}


_FLOAT_OK = re.compile(r"^(float|halo|orbit|spark|particle|aura|hover|ember|bubble|smoke|cloud)", re.I)


def components(parts: List[Dict[str, Any]], tol: float = 0.18) -> List[List[int]]:
    n, bx = len(parts), [aabb(p) for p in parts]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if all(bx[i][k][0] - tol <= bx[j][k][1] and bx[j][k][0] - tol <= bx[i][k][1] for k in range(3)):
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda c: (len(c), sum(vol(parts[i]) for i in c)), reverse=True)


def snap_floating(parts: List[Dict[str, Any]], max_gap: float = 16.0) -> int:
    snapped = 0
    for _ in range(7):
        comps = components(parts)
        if len(comps) <= 1:
            break
        bx = [aabb(p) for p in parts]
        main, moved = comps[0], False
        main_set = set(main)
        for comp in comps[1:]:
            if all(_FLOAT_OK.match(parts[i]["id"]) for i in comp):
                continue
            comp_prefix = "_".join(parts[comp[0]]["id"].split("_")[:2]).lower()
            candidates = [m for m in main if parts[m]["id"].lower().startswith(comp_prefix)] or main
            best = None
            for i in comp:
                for m in candidates:
                    d, mv = aabb_gap(bx[m], bx[i])
                    if best is None or d < best[0]:
                        best = (d, mv)
            if best is None or best[0] > 6.0:
                for i in comp:
                    for m in main_set:
                        d, mv = aabb_gap(bx[m], bx[i])
                        if best is None or d < best[0]:
                            best = (d, mv)
            limit = max_gap if len(comp) <= 6 else 9.0
            if best and best[0] <= limit:
                mv = [v + (0.12 if v > 0 else -0.12 if v < 0 else 0.0) for v in best[1]]
                for i in comp:
                    for k in range(3):
                        parts[i]["pos"][k] += mv[k]
                snapped += len(comp)
                moved = True
        if not moved:
            break
    return snapped


def analyze(parts: List[Dict[str, Any]], cat: str, sub: str, lo: int, hi: int, prompt: str,
            overlap: Dict[str, Any], snapped: int, face_relieved: int, rig_welded: int) -> Dict[str, Any]:
    n, issues, score = len(parts), [], 100.0
    glo, ghi = bounds(parts)
    ext = [ghi[k] - glo[k] for k in range(3)]
    gate = int(lo * GATE)

    def add(code: str, msg: str, blocking: bool, penalty: float) -> None:
        nonlocal score
        issues.append({"code": code, "msg": msg, "blocking": blocking})
        score -= penalty

    if n < lo:
        add("PART_COUNT_LOW", f"only {n} parts; the target is {lo}-{hi}. Add 4-layer facial details (brows, glowing eyes, open mouth cavity, 6 white fangs, pink tongue), sculpted chest plates, wrist bracers, and individual fingers/toes.",
            n < gate, min(35.0, (lo - n) / lo * 60))

    comps = components(parts)
    stray = [c for c in comps[1:] if not all(_FLOAT_OK.match(parts[i]["id"]) for i in c)]
    if stray:
        names = ", ".join(parts[c[0]]["id"] for c in stray[:4])
        add("FLOATING", f"{len(stray)} group(s) of parts are not connected to the main body (e.g. {names}). Move them so they overlap a neighbour.",
            True, min(25.0, 10.0 * len(stray)))
    float_msg = (f"{len(stray)} disconnected group(s) left." if stray else
                 f"Every part is connected ({snapped} stitched, {rig_welded} rigging joints smoothed, {face_relieved} facial features extruded in 3D relief).")

    if overlap["remaining"]:
        add("OVERLAP", f"{overlap['remaining']} pair(s) of parts still share the exact same surface.", False, min(10.0, 2.5 * overlap["remaining"]))
    if overlap["pairs"]:
        shown = "; ".join(overlap["names"][:2])
        overlap_msg = (f"{overlap['pairs']} pair(s) of parts shared a surface ({shown}) — automatically offset by {PUSH / 2:.2f} stud so nothing flickers in Studio.")
    else:
        overlap_msg = "No same-surface overlaps - nothing will flicker in Studio."

    prop_msg = "Proportions, rigging flow, and 3D facial relief look balanced."
    if cat in ("creature", "humanoid"):
        eyes = [p for p in parts if re.search(r"(?:^|_)eye(?:_|$|\d)", p["id"], re.I)]
        mouth_parts = [p for p in parts if re.search(r"mouth|jaw|fang|tooth|tongue|beak", p["id"], re.I)]
        if not eyes:
            add("MISSING_EYES", "the model has no visible Eye parts! Add glowing Eye_L, Eye_R, Pupil_L, Pupil_R and Brow blocks on the front of the head (+Z).", True, 15)
        if cat == "creature" and len(mouth_parts) < 3:
            add("MISSING_FACE_DETAIL", "the creature's face is missing an open roaring mouth! Add Head_Muzzle_Upper, Head_Jaw_Lower, dark Head_Mouth_Cavity, pink Head_Tongue, and 4-6 white Fang/Tooth blocks protruding at +Z.", True, 14)

    color_counts: Dict[str, int] = {}
    for p in parts:
        bucket = "#%x%x%x" % (int(p["color"][1:3], 16) >> 5, int(p["color"][3:5], 16) >> 5, int(p["color"][5:7], 16) >> 5)
        color_counts[bucket] = color_counts.get(bucket, 0) + 1
    distinct_cols = len(color_counts)
    max_col_share = max(color_counts.values()) / max(n, 1) if color_counts else 1.0
    if distinct_cols < 4 or max_col_share > 0.72:
        add("MONOCHROME", f"the coloring is too monochrome ({distinct_cols} distinct color bucket(s)). Use cream/tan `secondary` on chest/belly/muzzle/fingers/toes, gold `trim` on bracers/crown, bright `accent` on eyes, and white `light` on teeth/claws!",
            True, 16)

    avg_smooth = sum(p["smooth"] for p in parts) / max(n, 1)
    tilted = sum(1 for p in parts if lean_deg(p) > 2)

    score = max(0, int(round(score)))
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D"
    return {
        "issues": issues, "score": score, "grade": grade,
        "ext": ext, "tilted": tilted, "colors": distinct_cols,
        "diag": {
            "readout": f"{n} parts · {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} studs",
            "overlap_check": overlap_msg,
            "proportion_check": prop_msg,
            "floating_check": float_msg,
            "detail_check": f"{n} parts (target {lo}-{hi}), avg bevel smoothness {avg_smooth:.2f}, {distinct_cols} contrasting color zones.",
        },
    }


def process_build(data: Dict[str, Any], cat: str, sub: str, lo: int, hi: int, prompt: str, unit: float = 1.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = data.get("ops") if isinstance(data.get("ops"), list) else data.get("parts")
    if not isinstance(raw, list) or not raw:
        raise ValueError("The AI returned no parts.")
    if not isinstance(data.get("plan"), dict):
        data["plan"] = {}
    if not isinstance(data["plan"].get("palette"), dict):
        data["plan"]["palette"] = {}
    palette = data["plan"]["palette"]
    base = str(palette.get("base", "#18202c"))
    parts = normalize_ops(raw, unit, base if re.fullmatch(r"#[0-9a-fA-F]{6}", base) else "#18202c", cat, sub)
    if len(parts) < 3:
        raise ValueError("The AI returned fewer than 3 usable parts.")
    enforce_roblox_budget(parts, cat)
    limit_lean(parts)
    enforce_color_blocking(parts, palette, prompt, cat)
    parts = dedupe(parts)
    frame(parts)
    rig_welded = smooth_and_weld_rigging(parts, cat)
    stitched = stitch_articulated_chains(parts)
    face_relieved = extrude_facial_hierarchy(parts, cat)
    ov1 = fix_coplanar(parts)
    snapped = snap_floating(parts) + stitched
    frame(parts)
    ov2 = fix_coplanar(parts)
    overlap = {"pairs": ov1["pairs"] + ov2["pairs"], "names": ov1["names"] + ov2["names"], "remaining": ov2["remaining"]}
    return parts, analyze(parts, cat, sub, lo, hi, prompt, overlap, snapped, face_relieved, rig_welded)


# --------------------------------------------------------------------------- output & tonal contrast
_NO_JITTER = re.compile(r"eye|pupil|lens|light|glass|window|flame|glow|ember|spark|tooth|fang|tongue|crown|gem", re.I)


def jitter(color: str, key: str, amt: float = 0.045) -> str:
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if l < 0.12:
        l = 0.12 + l * 0.5
    delta = ((zlib.crc32(key.encode()) % 1000) / 1000 - 0.5) * 2 * amt
    l = max(0.08, min(0.95, l + delta))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def export_build(parts: List[Dict[str, Any]], analysis: Dict[str, Any], tonal: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ext = analysis["ext"]
    k = min(STUD, SCENE_MAX_M / max(max(ext), 1e-6))
    ops = []
    for p in parts:
        color = p["color"] if (not tonal or _NO_JITTER.search(p["id"])) else jitter(p["color"], p["id"])
        ops.append({
            "op": "add", "id": p["id"], "group": group_of(p["id"]), "shape": p["shape"],
            "size": [round(max(0.05, min(8.0, v * k)), 3) for v in p["size"]],
            "pos": [round(max(-12.0, min(12.0, v * k)), 3) for v in p["pos"]],
            "rot": [round(v, 1) for v in p["rot"]], "color": color, "smooth": round(p["smooth"], 3),
        })
    stats = {
        "parts": len(parts), "studs": [round(v, 1) for v in ext], "scene_scale": round(k, 4),
        "readout": f"{len(parts)} parts \u00b7 {ext[0]:.1f} \u00d7 {ext[1]:.1f} \u00d7 {ext[2]:.1f} studs",
    }
    return ops, stats


def better(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ka = (-sum(1 for i in a["issues"] if i["blocking"]), a["score"])
    kb = (-sum(1 for i in b["issues"] if i["blocking"]), b["score"])
    return ka > kb


# --------------------------------------------------------------------------- pipeline
async def ask_and_parse(system: str, user: str, provider: str, model: str, api_key: str, retries: int = 1):
    last: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            text, engine = await call_llm(system, user, provider, model, api_key)
        except ProviderError as e:
            raise e.to_http()
        data = parse_llm_json(text)
        if isinstance(data.get("ops") or data.get("parts"), list) and (data.get("ops") or data.get("parts")):
            return data, engine
        last = ValueError("The AI answer contained no parts.")
    raise HTTPException(status_code=422, detail=str(last))


def response_payload(title: str, cat: str, data: Dict[str, Any], parts, analysis, tonal: bool,
                     engine: Dict[str, str], revisions: int, used_serper: bool) -> Dict[str, Any]:
    ops, stats = export_build(parts, analysis, tonal)
    diag = dict(analysis["diag"])
    diag["quality"] = (f"Grade {analysis['grade']} ({analysis['score']}/100)"
                       + (f" after {revisions} automatic revision{'s' if revisions != 1 else ''}" if revisions else ""))
    return {
        "title": str(data.get("title") or title)[:60], "category": cat, "ops": ops, "stats": stats,
        "diagnostics": diag,
        "quality": {"score": analysis["score"], "grade": analysis["grade"], "revisions": revisions,
                    "issues_left": [i["msg"] for i in analysis["issues"]]},
        "plan": data.get("plan") if isinstance(data.get("plan"), dict) else None,
        "engine": engine, "used_serper": used_serper,
    }


class GenerateRequest(BaseModel):
    prompt: str
    web_context: Optional[str] = ""
    use_serper: bool = True
    provider: Optional[str] = ""
    model: Optional[str] = ""
    api_key: Optional[str] = ""
    serper_key: Optional[str] = ""
    category: Optional[str] = ""
    target_parts: Optional[int] = None
    revise: Optional[bool] = None
    max_revisions: Optional[int] = None
    tonal_variation: bool = True


class ReviseRequest(BaseModel):
    instruction: str
    ops: List[Dict[str, Any]]
    prompt: Optional[str] = ""
    category: Optional[str] = ""
    scene_scale: Optional[float] = None
    provider: Optional[str] = ""
    model: Optional[str] = ""
    api_key: Optional[str] = ""
    tonal_variation: bool = True


class TestKeyRequest(BaseModel):
    provider: str
    api_key: str
    model: Optional[str] = ""


@app.post("/generate")
async def generate(req: GenerateRequest):
    prompt = req.prompt.strip()[:1000]
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")
    cat, sub, wings, matched = classify(prompt, req.category or "")
    lo, hi = part_range(cat, req.target_parts)
    web_ctx = clean_web_context(req.web_context or "")
    if not web_ctx and req.use_serper:
        web_ctx = await fetch_anatomical_research(prompt, req.serper_key or "")
    system = build_system_prompt(cat, sub, wings, lo, hi)
    user = build_user_prompt(prompt, cat, sub, lo, hi, web_ctx)

    started = time.monotonic()
    data, engine = await ask_and_parse(system, user, req.provider or "", req.model or "", req.api_key or "")
    if not matched and not (req.category or "") and str(data.get("category", "")).lower() in PART_RANGE:
        cat = str(data["category"]).lower()
        lo, hi = part_range(cat, req.target_parts)
    try:
        parts, analysis = process_build(data, cat, sub, lo, hi, prompt)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    revisions = 0
    do_revise = AUTO_REVISE if req.revise is None else req.revise
    rounds = MAX_REVISIONS if req.max_revisions is None else max(0, min(2, req.max_revisions))
    for _ in range(rounds if do_revise else 0):
        blocking = [i["msg"] for i in analysis["issues"] if i["blocking"]]
        if not blocking:
            break
        elapsed = time.monotonic() - started
        if elapsed > REVISE_BUDGET_S * (revisions + 1) / (revisions + 2):
            break
        try:
            d2, _eng = await ask_and_parse(system, build_revision_prompt(prompt, cat, lo, hi, parts, blocking),
                                           req.provider or "", req.model or "", req.api_key or "", retries=0)
            p2, a2 = process_build(d2, cat, sub, lo, hi, prompt)
        except (HTTPException, ValueError):
            break
        revisions += 1
        if better(a2, analysis):
            data, parts, analysis = d2, p2, a2
    return JSONResponse(content=response_payload(prompt.title(), cat, data, parts, analysis, req.tonal_variation,
                                                 engine, revisions, bool(web_ctx)))


@app.post("/revise")
async def revise(req: ReviseRequest):
    change = req.instruction.strip()[:1000]
    if not change or not req.ops:
        raise HTTPException(status_code=400, detail="instruction and ops are required.")
    prompt = (req.prompt or "the current model").strip()[:1000]
    cat, sub, wings, _ = classify(f"{prompt} {change}", req.category or "")
    lo, hi = part_range(cat, None)
    unit = 1.0 / (req.scene_scale if req.scene_scale and req.scene_scale > 0 else STUD)
    current = normalize_ops(req.ops, unit, "#18202c", cat, sub)
    if len(current) < 3:
        raise HTTPException(status_code=400, detail="ops contained fewer than 3 usable parts.")
    frame(current)
    system = build_system_prompt(cat, sub, wings, lo, hi)
    data, engine = await ask_and_parse(system, build_revision_prompt(prompt, cat, lo, hi, current, [], change),
                                       req.provider or "", req.model or "", req.api_key or "")
    try:
        parts, analysis = process_build(data, cat, sub, lo, hi, prompt)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return JSONResponse(content=response_payload(prompt.title(), cat, data, parts, analysis, req.tonal_variation, engine, 1, False))


@app.get("/skill")
async def skill(prompt: str = "", category: str = ""):
    cat, sub, wings, _ = classify(prompt, category)
    lo, hi = part_range(cat, None)
    return {"category": cat, "sub": sub, "wings": wings, "parts": [lo, hi], "system_prompt": build_system_prompt(cat, sub, wings, lo, hi)}


@app.post("/test-key")
async def test_key(req: TestKeyRequest):
    prov, key, model = req.provider.lower().strip(), req.api_key.strip(), (req.model or "").strip()
    if not key:
        raise HTTPException(status_code=400, detail="API key is empty.")
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            if prov == "serper":
                res = await client.post(
                    "[https://google.serper.dev/search](https://google.serper.dev/search)",
                    headers={"X-API-KEY": key, "Content-Type": "application/json"},
                    json={"q": "3D low poly model", "num": 1},
                )
                if res.status_code != 200:
                    raise HTTPException(status_code=400, detail="Serper rejected key.")
                return {"ok": True, "message": "Serper verified & active!"}
            if prov == "gemini":
                m = normalize_gemini_model(model or "gemini-3.8-flash")
                res = await client.post(
                    f"[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){m}:generateContent",
                    headers={"x-goog-api-key": key},
                    json={"contents": [{"role": "user", "parts": [{"text": "OK"}]}], "generationConfig": {"maxOutputTokens": 16}},
                )
                if res.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Gemini error: {_err_text(res)}")
                return {"ok": True, "message": f"Gemini verified ({m})!"}
            if prov in ("openai", "groq"):
                base = "[https://api.openai.com/v1](https://api.openai.com/v1)" if prov == "openai" else "[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)"
                m = model or ("gpt-4o-mini" if prov == "openai" else DEFAULT_MODELS["groq"])
                payload: Dict[str, Any] = {"model": m, "messages": [{"role": "user", "content": "OK"}]}
                payload["max_completion_tokens" if _is_reasoning(m) else "max_tokens"] = 64 if _is_reasoning(m) else 10
                await _post_healing(client, prov, base + "/chat/completions", {"Authorization": f"Bearer {key}"}, payload)
                return {"ok": True, "message": f"{prov.title()} verified!"}
            if prov == "anthropic":
                await _post_healing(
                    client,
                    "anthropic",
                    "[https://api.anthropic.com/v1/messages](https://api.anthropic.com/v1/messages)",
                    {"x-api-key": key, "anthropic-version": "2023-06-01"},
                    {"model": model or DEFAULT_MODELS["anthropic"], "max_tokens": 10, "messages": [{"role": "user", "content": "OK"}]},
                )
                return {"ok": True, "message": "Anthropic verified!"}
            raise HTTPException(status_code=400, detail="Unknown provider")
        except HTTPException:
            raise
        except ProviderError as e:
            raise HTTPException(status_code=400, detail=f"{e.provider} rejected the key: {e.message}"[:1000])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "v18", "providers": [k for k, v in ENV_KEYS.items() if v],
            "auto_revise": AUTO_REVISE, "max_revisions": MAX_REVISIONS, "thinking_level": THINKING_LEVEL}


def find_ui() -> Optional[Path]:
    here = Path(__file__).parent
    if (here / "index.html").exists():
        return here / "index.html"
    found = sorted(here.glob("Verdant_3D*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = find_ui()
    if html_path:
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h3>index.html not found - put the Verdant page next to server.py</h3>")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
