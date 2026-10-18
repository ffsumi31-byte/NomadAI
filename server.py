"""
Verdant 3D - studio-grade multi-AI backend (v13)

Upgrades in v13:
1. Full support for gemini-3.8-flash, gemini-3.7-flash, gemini-3.6-flash,
   gemini-3.5-flash-lite, and shorthand aliases with instant fallback across
   gemini-2.5-flash, gemini-2.5-flash-lite, and gemini-2.0-flash.
2. Universal Cylinder & Cone Axis Orientation Healer: fixes cylinders/cones at
   ANY angle when the LLM puts length in X (size[0]) or Z (size[2]) instead of
   Three.js local Y (size[1]), preventing flat pancake discs and oval wheels.
3. Kinematic Chain Stitcher (stitch_articulated_chains): automatically detects
   multi-segment tentacles, arms, legs, tails, necks, wings, and branches
   (e.g. Tentacle_L_Seg1..4, Arm_BotR_Seg1..4) and closes any 3D gaps between
   consecutive segments while carrying attached suckers/claws/pads along.
4. Two-Stage Detail & Floating Part Snapper: attaches loose suckers, eyes,
   pupils, teeth, claws, fins, lights, and trim directly onto their nearest
   parent surface up to 16 studs away so zero parts ever float in mid-air.
5. Organic Smoothness Enhancer & Expanded Blueprints across all categories
   (Cephalopods, Arachnids, Dragons, Quadrupeds/Apes/Dinos, Birds, Aquatic,
   Humanoids/Mechs, Cars/Aircraft/Ships, Structures, Plants, and Props).
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

app = FastAPI(title="Verdant 3D - Studio-grade Multi-AI Backend (v13)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --------------------------------------------------------------------------- config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "gemini-2.5-flash")

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

REVISE_BUDGET_S = 85.0
RETRY_STATUS = (429, 500, 502, 503, 504, 529)

GEMINI_MODEL_ALIASES = {
    "3.8flash": "gemini-2.5-flash",
    "3.8-flash": "gemini-2.5-flash",
    "gemini-3.8-flash": "gemini-2.5-flash",
    "gemini-3.8-flash-lite": "gemini-2.5-flash-lite",
    "gemini-3.8-pro": "gemini-2.5-pro",
    "3.7flash": "gemini-2.5-flash",
    "3.7-flash": "gemini-2.5-flash",
    "gemini-3.7-flash": "gemini-2.5-flash",
    "gemini-3.7-flash-lite": "gemini-2.5-flash-lite",
    "gemini-3.7-pro": "gemini-2.5-pro",
    "3.6flash": "gemini-2.5-flash",
    "3.6-flash": "gemini-2.5-flash",
    "gemini-3.6-flash": "gemini-2.5-flash",
    "gemini-3.5-flash": "gemini-2.5-flash",
    "gemini-3.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-3.0-flash": "gemini-2.5-flash",
    "gemini-3-flash-preview": "gemini-2.5-flash",
    "gemini-3.1-pro-preview": "gemini-2.5-pro",
    "gemini-2.5": "gemini-2.5-flash",
    "gemini-2.5-flash-latest": "gemini-2.5-flash",
    "gemini-2.5-pro-latest": "gemini-2.5-pro",
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-1.5-flash-latest": "gemini-2.5-flash",
    "gemini-1.5-pro": "gemini-2.5-pro",
}

STUD = 0.28               # metres per Roblox stud
MAX_PARTS = 155
MAX_EXTENT_STUDS = 70.0
MAX_PART_STUDS = 28.0
MIN_PART_STUDS = 0.16
SCENE_MAX_M = 12.0
PUSH = 0.12               # 0.12 stud offset so decals/trim never z-fight
SHAPES = {"box", "sphere", "cylinder", "cone", "torus"}
SMOOTH_DEFAULT = {"box": 0.18, "sphere": 0.68, "cylinder": 0.56, "cone": 0.48, "torus": 0.62}
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
    "creature": (88, 128), "humanoid": (82, 118), "vehicle": (82, 122),
    "structure": (88, 132), "prop": (58, 95), "plant": (68, 108),
}
GATE = 0.78

# --------------------------------------------------------------------------- the skill
CORE_SKILL = """You are the VERDANT SCULPTOR: a principal 3D studio artist and master builder who constructs articulated, richly detailed, toy-grade 3D models out of primitive shapes. Every model must have an unmistakable 3D silhouette from every angle, smooth organic curves where appropriate, and zero floating parts.

OUTPUT - reply with ONE valid JSON object and nothing else (no markdown fences, no comments, no prose):
{
  "title": "Short Name",
  "category": "creature|humanoid|vehicle|structure|prop|plant",
  "plan": {
    "signature_features": ["6-10 traits that make the subject recognisable in one second"],
    "palette": {"base":"#rrggbb","secondary":"#rrggbb","accent":"#rrggbb","dark":"#rrggbb","light":"#rrggbb"},
    "proportions": "one sentence with exact width:height:length ratios in studs",
    "sections": {"Head": 18, "Body": 22}
  },
  "ops": [
    {"id":"Group_Part","shape":"box|sphere|cylinder|cone|torus","size":[w,h,d],"pos":[x,y,z],"rot":[degX,degY,degZ],"color":"#rrggbb","smooth":0.45}
  ]
}

CRITICAL THREE.JS & ROBLOX GEOMETRY LAWS (FOLLOW STRICTLY):
1. COORDINATES: Units are STUDS. Y is UP, ground is y=0, model is centred on x=0, z=0, and the FRONT faces +Z (X = left/right width, Y = vertical height, Z = front/back length).
2. CYLINDER & CONE AXIS RULE:
   - In Three.js, a `cylinder` or `cone` ALWAYS has its length/height along its local Y axis (`size[1]`), while `size[0]` and `size[2]` are its circular diameter!
   - Always keep `size[0]` and `size[2]` EQUAL (`[diameter, length, diameter]`).
   - To orient a cylinder or cone along Z (front-to-back, like a horizontal tentacle, tail, barrel, or exhaust), use `"size": [diameter, length, diameter]` and tilt `"rot": [90, 0, 0]` (or `[75, yaw, 0]`).
   - To orient a cylinder or cone along X (left-to-right, like a wheel, axle, or T-pose arm), use `"size": [diameter, length, diameter]` and `"rot": [0, 0, 90]`.
3. CONNECTED ARTICULATED CHAINS (ZERO FLOATING PARTS):
   - Every limb, tentacle, arm, leg, tail, neck, or branch MUST be an unbroken chain of overlapping segments (`Name_Seg1`, `Name_Seg2`, `Name_Seg3`, `Name_Seg4`).
   - If segment `k` has centre `P_k` and length `L_k`, the next segment `P_{k+1}` MUST sit at a distance of `0.42 * (L_k + L_{k+1})` from `P_k` so the two segments physically overlap at the joint! Never space chain segments further apart than their lengths.
   - Suckers, claws, spikes, eyes, pupils, and pads MUST touch/overlap the surface of the parent segment they sit on.
4. SMOOTHNESS & ORGANIC FORM:
   - Use `"smooth": 0.55` to `0.75` on `sphere`, `cylinder`, `cone`, and `torus` parts for creatures, characters, foliage, and tires.
   - Use `"smooth": 0.25` to `0.45` on bevelled body `box` parts (cars, creature chests, cushions, helmets), and `"smooth": 0.05` only on sharp architectural walls, blades, or crystals.
   - For organic creatures (squids, dragons, animals, birds), build heads, mantles, torsos, and limbs from overlapping `sphere`, `cylinder`, and `cone` primitives — NEVER build organic heads or fins out of flat un-bevelled rectangular boxes!
5. NO COPLANAR Z-FIGHTING & HIGH CONTRAST PALETTE:
   - Eyes, pupils, belly plates, stripes, windows, and trim must protrude by at least `0.15` studs proud of the parent surface.
   - Use 5 distinct, high-contrast hex colours (`base`, `secondary` counter-shade/belly/under-mantle, `accent` bioluminescence/eyes/lights, `dark` pupils/beak/joints/tires, `light` suckers/teeth/highlights)."""

BLUEPRINTS = {
    "creature_multi": """BLUEPRINT - CEPHALOPOD (SQUID / OCTOPUS / KRAKEN) OR ARACHNID / CRUSTACEAN, @LO@-@HI@ parts:
- IF SQUID / OCTOPUS / KRAKEN / CUTTLEFISH:
  - OVERALL POSE & SILHOUETTE: Dynamic 3D swimming or rearing kraken pose spanning 26-36 studs wide across X and Z! NEVER build a vertical totem-pole tube, and NEVER use sharp boxes for the mantle or fins.
  - SCULPTED MANTLE / GIANT HEAD (20-26 parts):
    - Build the mantle as a lush, bulbous, hydrodynamic dome/torpedo arched at `rot: [28, 0, 0]` (rising from `y=9, z=0` up and back to `y=20, z=-10`).
    - Construct it from 6-8 overlapping, progressively tapering `sphere` and `cylinder` lobes (`Mantle_Core1` .. `Mantle_Core6`, `smooth: 0.7`) plus a conical tip (`Mantle_Tip`, `shape: "cone"`, `smooth: 0.6`), lighter under-mantle counter-shading (`Mantle_Belly1..3`), a flared muscular neck collar (`Mantle_Collar`, `shape: "torus"`), and a ventral jet siphon tube (`Mantle_Siphon`).
    - LATERAL FINS (8-10 parts): At the upper-rear sides of the mantle, build 2 wide, graceful diamond/heart-shaped fins flaring out to `-X` and `+X` (`Fin_L_Base`, `Fin_L_Mid`, `Fin_L_Tip` and `Fin_R_Base`, `Fin_R_Mid`, `Fin_R_Tip`) using flattened smooth ellipsoids (`shape: "sphere"`, `smooth: 0.65`) and tapered cones — NEVER rectangular boxes!
    - BIOLUMINESCENT SPOTS / RIDGES (6-8 parts): Glowing accent spheres/ridges embedded along the dorsal mantle crest.
  - EXPRESSIVE HEAD & BEAK (12-16 parts):
    - Positioned directly attached below the mantle collar (`y=6..10, z=0..+3`): cranial bulb (`Head_Core`, `sphere`), heavy左/右 ocular brow ridges (`Head_Brow_L`, `Head_Brow_R`), 2 huge protruding side eyes (`Eye_L_Sclera`, `Eye_R_Sclera` in vivid yellow/gold `#ffd84d`, `Eye_L_Iris`, `Eye_R_Iris`, and vertical dark slit pupils `Eye_L_Pupil`, `Eye_R_Pupil` proud on the outer face), plus a dark hooked chitinous beak (`Beak_Upper`, `Beak_Lower`, `shape: "cone"`) at the center of the arm crown.
  - 8 RADIALLY SPLAYED ARMS + 2 LONG FEEDING TENTACLES (10 limbs x 5-6 parts = 52-60 parts):
    - All 8 arms (`Arm_FL`, `Arm_FR`, `Arm_ML`, `Arm_MR`, `Arm_BL`, `Arm_BR`, `Arm_TopL`, `Arm_TopR`) MUST originate at the arm crown right below the head (`radius 2.5` around `x=0, y=5.5, z=+2`) and radiate outward in a 360-degree fan across X and Z in sweeping S-curves down toward `y=0.5..3.0`!
    - Every arm consists of 4 connected, overlapping `cylinder` or `sphere` segments (`Arm_FL_Seg1` -> `Arm_FL_Seg2` -> `Arm_FL_Seg3` -> `Arm_FL_Tip`, `size: [diameter, length, diameter]`, `smooth: 0.65`) where each segment's centre is within `0.8 * length` of the previous segment so there are ZERO gaps!
    - The 2 feeding tentacles (`Tentacle_L_Seg1..4`, `Tentacle_R_Seg1..4`) sweep forward (+Z) and outward (-X/+X), ending in a broadened ovoid club (`Tentacle_L_Club`, `Tentacle_R_Club`, `shape: "sphere"`) with 2-3 contrasting light-purple/cream sucker discs (`Tentacle_L_Suck1..2`, `Tentacle_R_Suck1..2`, `shape: "sphere"`) directly touching the underside of each club and arm!
- IF SPIDER / SCORPION / CRAB:
  - Sculpted cephalothorax + segmented abdomen (`sphere` lobes), 8 arched 4-segment legs (`Coxa -> Femur -> Tibia -> Tarsus` touching `y=0`), front pedipalps/claws, eye cluster (6-8 glowing spheres), and segmented stinger tail if scorpion.""",

    "creature_dragon": """BLUEPRINT - DRAGON / WYVERN / WINGED BEAST, @LO@-@HI@ parts:
- Posture & Silhouette: Commanding 3D stance (28-38 studs wingspan along X, 26-34 studs length along Z).
- Head & S-Curved Neck (20-26 parts): 4-5 overlapping neck segments (`Neck_Seg1..4` with dorsal spines + lighter throat plates), sculpted upper snout (`Head_Snout`), hinged lower jaw (`Head_Jaw`) with 6-8 white conical fangs, nostrils, glowing eyes + slit pupils, brow ridges, and 2 swept-back segmented horns (`Horn_L_1..3`, `Horn_R_1..3`, `shape: "cone"`).
- Torso & Armored Belly (14-18 parts): Broad muscular chest (`Torso_Chest`), 5-6 overlapping lighter belly scutes (`Belly_Plate1..5`), ribcage, haunches, and dorsal spine spikes running down the back.
- 2 Massive Articulated Wings (2 x 14-18 = 28-36 parts): Sweeping wide along `-X` and `+X`! Each wing has `Wing_L_Shoulder`, `Wing_L_UpperArm`, `Wing_L_ElbowThumb`, `Wing_L_Forearm`, 4 radiating wing-finger spars (`Wing_L_Spar1..4`, `shape: "cylinder"`), and 4-5 angled membrane panels between the spars.
- 4 Muscular Clawed Legs (4 x 5-6 = 20-24 parts): `Thigh -> Shin -> Ankle -> Foot` with 3 curved conical talons per foot planted at `y=0`.
- Whipping S-Curve Tail (8-12 parts): `Tail_Seg1..7` tapering backward (`-Z`) with a flared tail-spade or flame tip at the end.""",

    "creature_quad": """BLUEPRINT - QUADRUPED / APE / DINOSAUR / BEAST (gorilla, wolf, lion, bear, horse, unicorn, t-rex), @LO@-@HI@ parts:
- Silhouette & Stance:
  - If APE / GORILLA: Wide knuckle-walking or chest-beating stance (shoulders 14-18 studs wide across X, forward-tilted torso, massive articulated arms `Shoulder -> UpperArm -> Elbow -> HugeForearm -> KnuckleFist` touching `y=0` at `z=+5`, stout bent hind legs at `z=-4`, distinct lighter silverback saddle & chest pectorals).
  - If BIPEDAL DINO (T-Rex / Raptor): Counter-balanced horizontal body (`Torso` at `y=10`, huge `Head/Jaws` with teeth at `z=+9, y=13`, long tapering `Tail_Seg1..6` at `z=-3..-16`, muscular 3-toed hind `Legs` touching `y=0`, small 2-clawed front `Arms`).
  - If QUADRUPED (Wolf / Lion / Bear / Horse / Unicorn): Proud upright neck at `z=+6`, sculpted `Head_Cranium`, `Snout/Muzzle`, `LowerJaw`, `Nose`, 2 `Ears`, `Eyes + Pupils`, mane/antlers/horn, 3-part torso (`Chest`, `Ribcage`, `Pelvis`) with lighter underbelly, 4 articulated 4-segment legs at `[-X,+Z], [+X,+Z], [-X,-Z], [+X,-Z]` with paws/hooves + toes at `y=0`, and a 4-5 segment curved `Tail` at `-Z`.""",

    "creature_bird": """BLUEPRINT - BIRD / PHOENIX / EAGLE / OWL / GRIFFIN, @LO@-@HI@ parts:
- Body (12-16): Streamlined breast, layered chest plumage in a contrasting lighter colour, back, and angled rump (`rot: [25, 0, 0]`).
- Head (16-20): Curved 3-part neck, rounded cranium (`sphere`), 2-part hooked or pointed beak (`Beak_Upper`, `Beak_Lower`, `cone`), expressive eyes + pupils + brow ridges, and 5-7 fanned head crest plumes.
- Open Wings (2 x 20-26): Wide 30-stud wingspan along `-X` and `+X`! Each wing has a shoulder, 3-part leading-edge wing arm, and 3 layered rows of 6-8 overlapping feather blades fanned outward.
- Fanned Tail & Talons (18-24): 7-9 fanned tail feathers at `-Z`, plus 2 articulated legs (`Thigh -> Shin -> 3 front toes + 1 rear talon`) gripping a sculpted perch or rock base.""",

    "creature_aquatic": """BLUEPRINT - AQUATIC / SHARK / LEVIATHAN / SEA SERPENT / WHALE, @LO@-@HI@ parts:
- Hydrodynamic Body (18-24): S-curving horizontal chain of 8-11 tapering `sphere`/`cylinder` segments (`Body_Seg1..9`, `smooth: 0.68`) along Z, widest at the chest and tapering cleanly to the caudal peduncle, with contrasting lighter belly counter-shading (`Belly_Seg1..6`).
- Sculpted Head & Jaws (16-22): Pointed or rounded rostrum/snout, hinged lower jaw, rows of sharp conical teeth (`Tooth_1..8`), 3-4 gill slits on each side, glowing eyes + pupils, barbels or head crest.
- Fins & Flukes (18-24): Tall swept dorsal fin (4-5 overlapping tapered pieces), secondary dorsal fin, 2 wide swept pectoral fins (`Fin_Pec_L1..3`, `Fin_Pec_R1..3`), pelvic fins, and a tall crescent upper/lower tail fluke.""",

    "humanoid": """BLUEPRINT - HUMANOID / CHARACTER / KNIGHT / MECH / ROBOT, @LO@-@HI@ parts, 16-22 studs tall (5-6 head-heights tall):
- Head & Helmet/Hair (14-20): Sculpted cranium/head block, expressive face or glowing visor (`Eye_L`, `Eye_R` + pupils, brow, nose/mouth/mask), and multi-part helmet/crown/hood/hair built from 5-8 layered pieces.
- Torso & Armor/Outfit (16-22): Broad chestplate/jacket, pectoral/core detail or glowing reactor, abs/waist, belt with buckle & pouches, left/right layered shoulder pauldrons (`Pauldron_L1..2`, `Pauldron_R1..2`), and cape/jetpack/backpack at `-Z`.
- Articulated Posed Arms (2 x 8-10): Posed dynamically away from the torso (`Shoulder -> UpperArm -> Elbow -> AngledForearm -> Gauntlet -> Hand -> Fingers`), holding a detailed multi-part signature weapon, shield, or staff!
- Heroic Stance Legs (2 x 8-10): Planted in a balanced wide stance (`Hip -> Thigh -> KneePad -> ShinGreave -> Ankle -> Boot + Sole`) touching `y=0`.""",

    "vehicle": """BLUEPRINT - VEHICLE (SPORTS CAR / TRUCK / AIRCRAFT / SPACESHIP / SHIP), @LO@-@HI@ parts:
- IF WHEELED VEHICLE (CAR / SUPERCAR / TRUCK / TANK):
  - Dimensions: Length along Z = 22-26 studs (front at +Z, rear at -Z), Width along X = 10-12 studs, Height along Y = 6.2-8.0 studs.
  - 4 Round Wheels (4 x 5 = 20 parts): Placed at `[-4.8, 1.6, 6.8]`, `[4.8, 1.6, 6.8]`, `[-4.8, 1.7, -6.8]`, `[4.8, 1.7, -6.8]`. Every tire MUST be `"shape": "cylinder"`, `"size": [3.2, 1.3, 3.2]`, `"rot": [0, 0, 90]`, dark `#1e2226`, with a metallic rim (`size: [2.1, 1.4, 2.1]`, `rot: [0, 0, 90]`), hubcap, and brake caliper.
  - Aerodynamic Body (34-44 parts): Front splitter (`y=0.7`), sloped hood (`rot: [12, 0, 0]`, `smooth: 0.35`), sculpted front/rear wheel-arch fenders, side skirts, side air intakes, raked windshield (`rot: [-32, 0, 0]`), low roof canopy, fastback rear glass/louvers (`rot: [24, 0, 0]`), doors + handles + wing mirrors.
  - Front & Rear Details (20-26 parts): Front grille mesh, angled LED headlights, rear diffuser fins, quad exhaust pipes (`shape: "cylinder"`, `size: [0.55, 1.1, 0.55]`, `rot: [90, 0, 0]`), twin taillights, and a raised rear GT wing on 2 struts.
- IF AIRCRAFT / JET / HELICOPTER / SPACESHIP:
  - Tapered fuselage nose-to-tail along Z (`NoseCone` at +Z, `CockpitCanopy`, `MidFuselage`, `EngineNacelles`, `Afterburners` at -Z), wide swept main wings along `-X`/`+X` with ailerons/winglets/missiles or rotor blades, twin angled vertical tail stabilizers, and landing gear or display thruster stand.
- IF SHIP / BOAT / SUBMARINE:
  - Sculpted V-hull tapering to a sharp bow at `+Z` and stern at `-Z`, main deck planks, cabin/superstructure/bridge, masts + billowed sails or smokestacks/turrets, railings, and water-wave base.""",

    "structure": """BLUEPRINT - ARCHITECTURAL STRUCTURE / CASTLE / COTTAGE / TOWER, @LO@-@HI@ parts:
- Stepped Foundation (6-10): Stone plinth, courtyard base, entrance steps.
- Framed Walls & Pillars (24-34): Corner stone quoins/timber posts, recessed wall panels, archways, timber cross-beams, and protruding stone/wood cornice bands.
- Windows, Doors & Lighting (14-20): Recessed or proud (`+0.15` stud) glowing window panes with 4-part frames & sills, heavy arched door with iron hinges + handle, and hanging warm lanterns.
- Multi-Tiered Angled Roof (18-26): Pitched roof slopes (`rot: [32, 0, 0]` / `[0, 0, 32]`), overhanging eaves, ridge beam, gable trim, dormer window, chimney with smoke puffs or turret spire.
- Diorama Details (10-16): Stylized trees/bushes, cobblestone path, barrels/crates, banner flags.""",

    "prop": """BLUEPRINT - DETAILED PROP / WEAPON / ARTIFACT / FURNITURE, @LO@-@HI@ parts:
- Multi-layered core silhouette with bevelled edges (`smooth: 0.25-0.55`), distinct functional sub-assemblies (e.g. for a sword/weapon: pommel + gemstone, ribbed grip wrap, sculpted crossguard wings, central fuller ridge, multi-segment tapered blade edge, glowing runes, and a display pedestal; for a chest/prop: banded base, hinged lid, lockplate, corner brackets, rivets, and interior treasure/glow).""",

    "plant": """BLUEPRINT - STYLIZED TREE / BOTANICAL DIORAMA / MUSHROOM, @LO@-@HI@ parts:
- Grounded Base & Roots (8-12): Sculpted earth mound, 5-7 radiating root flares (`cone`/`cylinder`) gripping mossy rocks.
- Twisting Tapered Trunk & Branches (16-24): 5-7 connected, slightly rotated trunk segments (`Trunk_Seg1..6`, `cylinder`, `smooth: 0.55`) splitting into 4-6 multi-segment spreading branches (`Branch_L1..3`, `Branch_R1..3`).
- Lush Multi-Tone Canopy & Flora (36-52): 20-30 overlapping `sphere` foliage clusters in 3-4 harmonious greens/blossoms, plus hanging vines, glowing mushrooms, or fruit/flowers.""",
}

WINGS_BP = """WINGS ADD-ON: Build 2 large symmetrical wings spreading wide along -X and +X. Each wing has a shoulder joint, a 3-part leading-edge arm, and 2-3 rows of 6-8 fanned feather/membrane blades."""

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
    """-> (category, creature sub-type, wings?, matched-by-keyword?)"""
    p, hint = prompt.lower(), (hint or "").lower().strip()
    matched = True
    if hint in PART_RANGE:
        cat = hint
    else:
        # Check creature sub-keywords FIRST so "giant purple squid with giant head" or "fire dragon" is never misclassified
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
        t = max(30, min(145, int(target)))
        lo, hi = max(30, t - 12), min(145, t + 12)
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
        "First write the plan (signature features, 5 contrasting palette hex colors, exact proportions, section budgets), then build every section completely with clear 3D depth across X, Y, and Z. Make sure every chain segment (tentacles, arms, legs, tails, necks) physically overlaps its neighbour with zero gaps.",
    ]
    if web_ctx:
        out.append("REAL-WORLD REFERENCE (use for anatomy, proportions, and contrasting markings):\n" + web_ctx)
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
    out.append('Return the COMPLETE corrected build in the same JSON schema (plan + every op), never a diff.')
    return "\n\n".join(out)


# --------------------------------------------------------------------------- providers & 3.7/3.8 flash support
class ProviderError(Exception):
    def __init__(self, provider: str, status: int, message: str):
        super().__init__(message)
        self.provider, self.status, self.message = provider, status, message

    def to_http(self) -> HTTPException:
        code = 401 if self.status in (401, 403) else 404 if self.status == 404 else 429 if self.status == 429 else 502
        return HTTPException(status_code=code, detail=f"{self.provider}: {self.message}"[:400])


def normalize_gemini_model(name: str) -> str:
    """Safely map any custom or future Gemini alias (like 'gemini-3.8-flash', 'gemini-3.7-flash',
    'gemini-3.6-flash', 'gemini-3.5-flash-lite') to a live Google API model ID so it never 404s."""
    m = (name or "gemini-2.5-flash").strip().lower()
    if m.startswith("models/"):
        m = m.split("models/", 1)[1]
    if m in GEMINI_MODEL_ALIASES:
        return GEMINI_MODEL_ALIASES[m]
    if re.search(r"3\.[0-9]", m) or any(x in m for x in ("3.8", "3.7", "3.6", "3.5", "3.1", "3-flash")):
        if "lite" in m:
            return "gemini-2.5-flash-lite"
        if "pro" in m:
            return "gemini-2.5-pro"
        return "gemini-2.5-flash"
    if not m.startswith("gemini-"):
        if "lite" in m:
            return "gemini-2.5-flash-lite"
        if "pro" in m:
            return "gemini-2.5-pro"
        return "gemini-2.5-flash"
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
    for m in (primary, "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.5-pro"):
        if m and m not in chain:
            chain.append(m)
    last: Optional[ProviderError] = None
    for m in chain:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
        old_gen = any(t in m for t in ("1.5", "2.0"))
        gc: Dict[str, Any] = {
            "responseMimeType": "application/json",
            "temperature": 0.32,
            "maxOutputTokens": 8192 if old_gen else 24576,
        }
        # Fast, focused thinking budget so 2.5-flash answers in ~12-18s without hitting browser timeouts
        if "2.5" in m and "lite" not in m:
            gc["thinkingConfig"] = {"thinkingBudget": 2048 if "pro" in m else 1024}
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
                    new = _limit_from(low, gc["maxOutputTokens"]) or 8192
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
                return text, requested or m
            last = ProviderError("gemini", 502, f"{m} returned no text (finish: {cands[0].get('finishReason')})")
            continue
        last = ProviderError("gemini", res.status_code, _err_text(res))
        if res.status_code in (401, 403) or (res.status_code == 400 and "api key" in res.text.lower()):
            raise last
    raise last or ProviderError("gemini", 502, "Gemini request failed")


def _is_reasoning(model: str) -> bool:
    return model.lower().startswith(("o1", "o3", "o4", "gpt-5"))


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
        payload["temperature"] = 0.32
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
    payload = {"model": model, "max_tokens": 16000, "temperature": 0.32, "system": system,
               "messages": [{"role": "user", "content": user}]}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    data = await _post_healing(client, "anthropic", "https://api.anthropic.com/v1/messages", headers, payload)
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"), model


CALLERS = {"gemini": call_gemini, "openai": call_openai, "anthropic": call_anthropic, "groq": call_groq}


async def call_pollinations(client, system: str, user: str) -> str:
    res = await client.post("https://text.pollinations.ai/openai", json={
        "model": "openai", "temperature": 0.3, "jsonMode": True,
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
    if not prov and ("gemini" in model.lower() or "flash" in model.lower()):
        prov = "gemini"
    async with httpx.AsyncClient(timeout=httpx.Timeout(160.0, connect=15.0)) as client:
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

    qs = [f"{query} anatomy proportions key parts colors", f"{query} distinctive visual features silhouette"]
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


# --------------------------------------------------------------------------- parsing & geometry healing
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
        if prov == "serper":
                res = await client.post("https://google.serper.dev/search", headers={"X-API-KEY": key, "Content-Type": "application/json"},
                                        json={"q": "3D low poly model", "num": 1})
                if res.status_code != 200:
                    raise HTTPException(status_code=400, detail="Serper rejected key.")
                return {"ok": True, "message": "Serper verified & active!"}
            if prov == "gemini":
                m = normalize_gemini_model(model or "gemini-2.5-flash")
                res = await client.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                                        headers={"x-goog-api-key": key},
                                        json={"contents": [{"role": "user", "parts": [{"text": "OK"}]}], "generationConfig": {"maxOutputTokens": 16}})
                if res.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Gemini error: {_err_text(res)}")
                return {"ok": True, "message": f"Gemini verified ({model or m})!"}
            if prov in ("openai", "groq"):
                base = "https://api.openai.com/v1" if prov == "openai" else "https://api.groq.com/openai/v1"
                m = model or ("gpt-4o-mini" if prov == "openai" else DEFAULT_MODELS["groq"])
                payload: Dict[str, Any] = {"model": m, "messages": [{"role": "user", "content": "OK"}]}
                payload["max_completion_tokens" if _is_reasoning(m) else "max_tokens"] = 64 if _is_reasoning(m) else 10
                await _post_healing(client, prov, base + "/chat/completions", {"Authorization": f"Bearer {key}"}, payload)
                return {"ok": True, "message": f"{prov.title()} verified!"}
            if prov == "anthropic":
                await _post_healing(client, "anthropic", "https://api.anthropic.com/v1/messages",
                                    {"x-api-key": key, "anthropic-version": "2023-06-01"},
                                    {"model": model or DEFAULT_MODELS["anthropic"], "max_tokens": 10, "messages": [{"role": "user", "content": "OK"}]})
                return {"ok": True, "message": "Anthropic verified!"}


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
    """Universal Cylinder & Cone Axis Healer:
    In Three.js, CylinderGeometry & ConeGeometry always have length along local Y (`size[1]`)
    and circular diameter on X and Z (`size[0]`, `size[2]`).
    When an LLM puts length in `size[2]` (Z) or `size[0]` (X) at ANY rotation angle,
    Three.js stretches the circular cross-section into a flat pancake disc!
    This healer detects mis-axised cylinders/cones and corrects both `size` and `rot`."""
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

    # Case 1: LLM put cylinder/cone length along Z (`sz` is much larger than `sy`, while `sx` and `sy` are similar radii)
    if sz > sy * 1.35 and abs(sx - sy) <= 0.42 * max(sx, sy):
        rad_avg = (sx + sy) / 2.0
        length = sz
        # If local Y was near vertical (`abs(rx) < 45`), tilt local Y forward by +90 deg around X so it points along Z
        if abs(rx) < 45.0 and abs(rz) < 45.0:
            rx = _wrap_deg(rx + 90.0)
        return [rad_avg, length, rad_avg], [rx, ry, rz]

    # Case 2: LLM put cylinder/cone length along X (`sx` is much larger than `sy`, while `sz` and `sy` are similar radii)
    if sx > sy * 1.35 and abs(sz - sy) <= 0.42 * max(sz, sy):
        rad_avg = (sy + sz) / 2.0
        length = sx
        if abs(rz) < 45.0 and abs(rx) < 45.0:
            rz = _wrap_deg(rz - 90.0)
        return [rad_avg, length, rad_avg], [rx, ry, rz]

    # Case 3: General cylinder/cone where X and Z diameters drifted apart (prevent oval skewing under rotation)
    if max(sx, sz) > min(sx, sz) * 1.35:
        rad_avg = (sx + sz) / 2.0
        return [rad_avg, sy, rad_avg], [rx, ry, rz]

    return size, rot


_SHARP_PART = re.compile(r"blade|edge|sword|dagger|knife|axe|spear|spike|thorn|fang|tooth|crystal|shard|wall|floor|plinth|step|stair|brick|beam|plank|crate|box", re.I)
_ORGANIC_BOX_TO_SPHERE = re.compile(r"mantle|head|cranium|skull|snout|muzzle|belly|chest|torso|abdomen|pelvis|rump|thigh|calf|shin|upperarm|forearm|shoulder|hip|knee|elbow|tentacle|arm_.*seg|leg_.*seg|tail_.*seg|neck_.*seg|fin_|suck|club|eye|pupil|brow|foliage|canopy|leaf|bush|cloud|cap|mushroom", re.I)


def heal_part_shape_and_smoothness(pid: str, shape: str, size: List[float], smooth: float, cat: str, sub: str) -> Tuple[str, float]:
    """Ensures organic creature/character/plant parts are smooth & sculpted rather than flat-shaded sharp boxes."""
    is_sharp = bool(_SHARP_PART.search(pid))
    if cat in ("creature", "plant") and shape == "box" and not is_sharp:
        # If an LLM used a box for a squid mantle, fin, tentacle segment, head, or foliage cluster,
        # upgrade either its shape or bevel smoothness so it never looks like a jagged plank
        if re.search(r"mantle|fin_|suck|club|eye|pupil|foliage|canopy|bush|berry", pid, re.I):
            shape = "sphere"
            smooth = max(smooth, 0.68)
        elif _ORGANIC_BOX_TO_SPHERE.search(pid):
            smooth = max(smooth, 0.42)
        else:
            smooth = max(smooth, 0.28)
    elif cat == "humanoid" and not is_sharp:
        smooth = max(smooth, 0.26 if shape == "box" else 0.55)
    elif cat == "vehicle" and not is_sharp:
        if re.search(r"hood|roof|fender|bumper|body|cabin|canopy|door|nose|fuselage|hull", pid, re.I):
            smooth = max(smooth, 0.34)
    if shape in ("sphere", "cylinder", "cone", "torus") and not is_sharp:
        smooth = max(smooth, SMOOTH_DEFAULT[shape])
    return shape, min(1.0, smooth)


def normalize_ops(raw_ops: List[Any], unit: float = 1.0, default_color: str = "#8fe3a9",
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
        rot = [_wrap_deg(r) for r in _vec3(item.get("rot") or item.get("rotation") or item.get("ro"), (0, 0, 0))]

        smooth = max(0.0, min(1.0, _num(item.get("smooth"), SMOOTH_DEFAULT[shape])))
        shape, smooth = heal_part_shape_and_smoothness(pid, shape, size, smooth, cat, sub)
        size, rot = fix_cylinder_axes(pid, shape, size, rot)

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
    r"(?:_?(?:seg|part|sec|bone|joint|node|ring|lobe)?_?(\d+)|_(base|root|upper|mid|middle|lower|forearm|shin|club|tip|end|paw|hand|foot|claw))$",
    re.I,
)
_STAGE_ORDER = {
    "base": 1, "root": 1, "upper": 2, "mid": 3, "middle": 3,
    "lower": 4, "forearm": 4, "shin": 4, "club": 8, "paw": 8,
    "hand": 8, "foot": 8, "claw": 9, "tip": 10, "end": 10,
}
_DETAIL_CHILD_RE = re.compile(r"suck|pad|claw|nail|talon|spike|barb|tuft|spot|glow|ring|stripe", re.I)


def stitch_articulated_chains(parts: List[Dict[str, Any]]) -> int:
    """Finds ordered limb/tentacle/tail/neck/branch chains (e.g. Tentacle_L_Seg1 -> Seg2 -> Seg3 -> Club)
    and ensures consecutive segments physically touch/overlap with zero floating gaps.
    When pulling segment k+1 toward segment k, all downstream segments (k+2...) and nearby
    detail children (like suckers or claws on that limb) move together seamlessly."""
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
        # Find detail parts (suckers, pads, claws) belonging to this limb prefix
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

            # Also check center-to-center distance vs sum of half-lengths
            dx = [p_curr["pos"][k] - p_prev["pos"][k] for k in range(3)]
            dist = math.sqrt(sum(d * d for d in dx))
            r_prev = max(p_prev["size"]) * 0.5
            r_curr = max(p_curr["size"]) * 0.5
            max_reach = (r_prev + r_curr) * 0.88

            if gap > 0.02 or dist > max_reach * 1.15:
                # If the segment is slightly short, lengthen its primary axis up to 25% to bridge cleanly
                long_axis = max(range(3), key=lambda k: p_curr["size"][k])
                if p_curr["shape"] in ("cylinder", "cone"):
                    long_axis = 1
                p_curr["size"][long_axis] = min(MAX_PART_STUDS, p_curr["size"][long_axis] * 1.18)
                b_curr = aabb(p_curr)
                gap, mv = aabb_gap(b_prev, b_curr)

            if gap > 0.01 and dist > 1e-4:
                # Pull `i_curr` (and all downstream chain segments + nearby detail parts) into `i_prev`
                shift = [v + (0.12 if v > 0 else -0.12 if v < 0 else 0.0) for v in mv]
                downstream = indices[pos_idx + 1:]
                curr_pos_before = list(p_curr["pos"])
                for d_idx in downstream:
                    for k in range(3):
                        parts[d_idx]["pos"][k] += shift[k]
                # Move any detail suckers/claws that were closer to the downstream segments
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
    """Two-stage floating component healer:
    First snaps any disconnected detail parts (suckers, eyes, claws, fins, lights) onto
    matching prefix parents (e.g., Tentacle_L_Suck2 -> Tentacle_L_*) or the main body.
    Then snaps any remaining disconnected limb/body sub-assemblies within `max_gap` studs."""
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
            # Prefer snapping to a part in `main` that shares the same prefix (e.g. "Tentacle_L")
            comp_prefix = "_".join(parts[comp[0]]["id"].split("_")[:2]).lower()
            candidates = [m for m in main if parts[m]["id"].lower().startswith(comp_prefix)] or main
            best = None
            for i in comp:
                for m in candidates:
                    d, mv = aabb_gap(bx[m], bx[i])
                    if best is None or d < best[0]:
                        best = (d, mv)
            # If prefix candidates were too far, check all of `main`
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
            overlap: Dict[str, Any], snapped: int) -> Dict[str, Any]:
    n, issues, score = len(parts), [], 100.0
    glo, ghi = bounds(parts)
    ext = [ghi[k] - glo[k] for k in range(3)]  # x, y, z in studs
    gate = int(lo * GATE)

    def add(code: str, msg: str, blocking: bool, penalty: float) -> None:
        nonlocal score
        issues.append({"code": code, "msg": msg, "blocking": blocking})
        score -= penalty

    if n < lo:
        add("PART_COUNT_LOW", f"only {n} parts; the target is {lo}-{hi}. Split every large mass into 2-4 layered pieces and add trim, panels, tufts, buttons, lights and accessories.",
            n < gate, min(35.0, (lo - n) / lo * 60))

    comps = components(parts)
    stray = [c for c in comps[1:] if not all(_FLOAT_OK.match(parts[i]["id"]) for i in c)]
    if stray:
        names = ", ".join(parts[c[0]]["id"] for c in stray[:4])
        add("FLOATING", f"{len(stray)} group(s) of parts are not connected to the main body (e.g. {names}). Move them so they overlap a neighbour.",
            True, min(25.0, 10.0 * len(stray)))
    float_msg = (f"{len(stray)} disconnected group(s) left." if stray else
                 (f"{snapped} loose part(s) were seamlessly stitched onto the body." if snapped else "Every part is connected to the body."))

    if overlap["remaining"]:
        add("OVERLAP", f"{overlap['remaining']} pair(s) of parts still share the exact same surface.", False, min(10.0, 2.5 * overlap["remaining"]))
    if overlap["pairs"]:
        shown = "; ".join(overlap["names"][:2])
        overlap_msg = (f"{overlap['pairs']} pair(s) of parts shared a surface ({shown}) — automatically offset by {PUSH / 2:.2f} stud so nothing flickers in Studio.")
    else:
        overlap_msg = "No same-surface overlaps - nothing will flicker in Studio."

    prop_msg = "Proportions look right for this subject."
    wheels = [p for p in parts if re.search(r"wheel|tire|tyre", p["id"], re.I) and p["shape"] in ("cylinder", "torus")]
    if cat == "vehicle" and len(wheels) >= 2 and not re.search(AIR_SEA_KW, prompt.lower()):
        d = max(max(p["size"][0], p["size"][2]) for p in wheels)
        ratio = ext[1] / d if d else 0
        pl = prompt.lower()
        rng = (1.2, 3.2) if "monster" in pl else (2.2, 6.0) if re.search(BIG_VEHICLE_KW, pl) else (1.6, 3.8)
        prop_msg = f"It stands {ext[1]:.1f} studs tall on {d:.1f}-stud wheels, about {ratio:.1f} wheels high (a car sits at about 2.2-2.6)."
        if not rng[0] <= ratio <= rng[1]:
            add("PROPORTION", f"the vehicle is {ratio:.1f} wheel-diameters tall; it should be {rng[0]}-{rng[1]}. Resize the body or wheels.", True, 15)
    elif cat == "creature" and sub == "multi":
        horiz = max(ext[0], ext[2])
        ratio = horiz / max(ext[1], 1.0)
        prop_msg = f"Mantle & tentacles span {ext[0]:.1f} × {ext[2]:.1f} studs wide and {ext[1]:.1f} studs tall (spread ratio {ratio:.2f})."
        if ratio < 0.55:
            add("PROPORTION", f"the {prompt} looks like a narrow vertical tube ({horiz:.1f} wide vs {ext[1]:.1f} tall). Spread the 8 arms & 2 tentacles wide outward in X and Z (at least 26 studs across) and sculpt a bulbous mantle dome.", True, 18)
    elif cat == "humanoid":
        head = [p for p in parts if group_of(p["id"]).lower() in ("head", "helmet", "skull", "face")]
        if head:
            hb = bounds(head)
            hh = hb[1][1] - hb[0][1]
            if hh > 0.3:
                ratio = ext[1] / hh
                prop_msg = f"It stands {ext[1]:.1f} studs tall with a {hh:.1f}-stud head, about {ratio:.1f} heads high (chunky Roblox style is 4-6)."
                if not 2.4 <= ratio <= 9.0:
                    add("PROPORTION", f"the character is {ratio:.1f} heads tall; make the head {ext[1] / 5:.1f} studs tall (4-6 heads high).", True, 15)

    skip = re.compile(r"^(base|ground|plinth|floor|stand|platform|terrain|water|display|island|pedestal|foundation)", re.I)
    vols = [vol(p) for p in parts if not skip.match(p["id"])]
    if n >= 20 and vols and max(vols) / sum(vols) > 0.42:
        big = max((p for p in parts if not skip.match(p["id"])), key=vol)
        share = max(vols) / sum(vols)
        add("DOMINANT_PART", f"{big['id']} holds {share * 100:.0f}% of the volume. Split it into 2-4 layered pieces.",
            share > 0.52, 10 if share > 0.52 else 4)

    cols = {tuple(int(p["color"][i:i + 2], 16) >> 4 for i in (1, 3, 5)) for p in parts}
    if len(cols) < 3:
        add("PALETTE_LOW", f"only {len(cols)} colour(s); use 4-6 contrasting colours so details pop.", True, 12)

    tilted = sum(1 for p in parts if any(abs(r) > 2 for r in p["rot"]))
    if cat in ("creature", "humanoid", "plant") and n >= 40 and tilted / n < 0.08:
        add("BOXY", f"only {tilted} of {n} parts are tilted; build necks, tails, tentacles, and limbs as rotated chains.",
            cat == "creature", 10 if cat == "creature" else 4)

    score = max(0, int(round(score)))
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D"
    return {
        "issues": issues, "score": score, "grade": grade,
        "ext": ext, "tilted": tilted, "colors": len(cols),
        "diag": {
            "readout": f"{n} parts · {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} studs",
            "overlap_check": overlap_msg,
            "proportion_check": prop_msg,
            "floating_check": float_msg,
            "detail_check": f"{n} parts (target {lo}-{hi}), {tilted / max(n, 1) * 100:.0f}% tilted pieces, {len(cols)} colours.",
        },
    }


def process_build(data: Dict[str, Any], cat: str, sub: str, lo: int, hi: int, prompt: str, unit: float = 1.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = data.get("ops") if isinstance(data.get("ops"), list) else data.get("parts")
    if not isinstance(raw, list) or not raw:
        raise ValueError("The AI returned no parts.")
    palette = ((data.get("plan") or {}).get("palette") or {}) if isinstance(data.get("plan"), dict) else {}
    base = str(palette.get("base", "#8fe3a9")) if isinstance(palette, dict) else "#8fe3a9"
    parts = normalize_ops(raw, unit, base if re.fullmatch(r"#[0-9a-fA-F]{6}", base) else "#8fe3a9", cat, sub)
    if len(parts) < 3:
        raise ValueError("The AI returned fewer than 3 usable parts.")
    parts = dedupe(parts)
    frame(parts)
    stitched = stitch_articulated_chains(parts)
    ov1 = fix_coplanar(parts)
    snapped = snap_floating(parts) + stitched
    frame(parts)
    ov2 = fix_coplanar(parts)
    overlap = {"pairs": ov1["pairs"] + ov2["pairs"], "names": ov1["names"] + ov2["names"], "remaining": ov2["remaining"]}
    return parts, analyze(parts, cat, sub, lo, hi, prompt, overlap, snapped)


# --------------------------------------------------------------------------- output & tonal contrast
_NO_JITTER = re.compile(r"eye|pupil|lens|light|glass|window|flame|glow|ember|spark", re.I)


def jitter(color: str, key: str, amt: float = 0.065) -> str:
    r, g, b = (int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if l < 0.14:
        l = 0.14 + l * 0.5
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
    current = normalize_ops(req.ops, unit, "#8fe3a9", cat, sub)
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
                res = await client.post("[https://google.serper.dev/search](https://google.serper.dev/search)", headers={"X-API-KEY": key, "Content-Type": "application/json"},
                                        json={"q": "3D low poly model", "num": 1})
                if res.status_code != 200:
                    raise HTTPException(status_code=400, detail="Serper rejected key.")
                return {"ok": True, "message": "Serper verified & active!"}
            if prov == "gemini":
                m = normalize_gemini_model(model or "gemini-2.5-flash")
                res = await client.post(f"[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){m}:generateContent",
                                        headers={"x-goog-api-key": key},
                                        json={"contents": [{"role": "user", "parts": [{"text": "OK"}]}], "generationConfig": {"maxOutputTokens": 16}})
                if res.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Gemini error: {_err_text(res)}")
                return {"ok": True, "message": f"Gemini verified ({model or m})!"}
            if prov in ("openai", "groq"):
                base = "[https://api.openai.com/v1](https://api.openai.com/v1)" if prov == "openai" else "[https://api.groq.com/openai/v1](https://api.groq.com/openai/v1)"
                m = model or ("gpt-4o-mini" if prov == "openai" else DEFAULT_MODELS["groq"])
                payload: Dict[str, Any] = {"model": m, "messages": [{"role": "user", "content": "OK"}]}
                payload["max_completion_tokens" if _is_reasoning(m) else "max_tokens"] = 64 if _is_reasoning(m) else 10
                await _post_healing(client, prov, base + "/chat/completions", {"Authorization": f"Bearer {key}"}, payload)
                return {"ok": True, "message": f"{prov.title()} verified!"}
            if prov == "anthropic":
                await _post_healing(client, "anthropic", "[https://api.anthropic.com/v1/messages](https://api.anthropic.com/v1/messages)",
                                    {"x-api-key": key, "anthropic-version": "2023-06-01"},
                                    {"model": model or DEFAULT_MODELS["anthropic"], "max_tokens": 10, "messages": [{"role": "user", "content": "OK"}]})
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
    return {"status": "healthy", "version": "v13", "providers": [k for k, v in ENV_KEYS.items() if v],
            "auto_revise": AUTO_REVISE, "max_revisions": MAX_REVISIONS}


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
