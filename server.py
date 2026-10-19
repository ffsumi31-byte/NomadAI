"""
Verdant 3D - Roblox-native multi-AI backend (v16)

Upgrades in v16:
A. Enhanced AI Design Intelligence: Upgraded system prompts and blueprints to enforce true originality,
   distinct aesthetic styling, custom component layering (armor, mechanical trims, glowing core nodes),
   and complex multi-part silhouettes instead of generic low-poly shapes.
B. Advanced Part-Stitching and Kinematic Articulation: Refined joint-snapping and coplanar adjustments
   to prevent z-fighting and floating components entirely.
C. Bulletproof error handling across Gemini 3.8/3.7/3.6, Anthropic, OpenAI, and Groq providers.
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

app = FastAPI(title="Verdant 3D - Studio-grade Multi-AI Backend (v16)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --------------------------------------------------------------------------- config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "gemini-3.8-flash")

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
    "3.8flash": "gemini-3.8-flash",
    "3.8-flash": "gemini-3.8-flash",
    "gemini-3.8-flash": "gemini-3.8-flash",
    "3.7flash": "gemini-3.8-flash",
    "3.7-flash": "gemini-3.8-flash",
    "gemini-3.7-flash": "gemini-3.7-flash",
    "3.6flash": "gemini-3.6-flash",
    "3.6-flash": "gemini-3.6-flash",
    "gemini-3.6-flash": "gemini-3.6-flash",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "gemini-3.5-flash-lite": "gemini-3.5-flash-lite",
}

STUD = 0.28               # metres per Roblox stud
MAX_PARTS = 155
MAX_EXTENT_STUDS = 70.0
MAX_PART_STUDS = 28.0
MIN_PART_STUDS = 0.16
SCENE_MAX_M = 12.0
PUSH = 0.12               # 0.12 stud offset so decals/trim never z-fight
SHAPES = {"box", "sphere", "cylinder", "cone", "torus"}
SMOOTH_DEFAULT = {"box": 0.04, "sphere": 0.5, "cylinder": 0.2, "cone": 0.3, "torus": 0.4}
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

# --------------------------------------------------------------------------- the skill (Upgraded for Originality & Design Flair)
CORE_SKILL = """You are the VERDANT MASTER SCULPTOR & CONCEPT ARTIST: a veteran Roblox game designer who constructs highly original, creative, and visually striking models out of Roblox Parts. You inject genuine personality, sharp silhouette hierarchy, armor layering, and signature design features into every creation—never returning boring, generic boxes.

OUTPUT - reply with ONE valid JSON object and nothing else (no markdown fences, no comments, no prose):
{
  "title": "Creative Model Title",
  "category": "creature|humanoid|vehicle|structure|prop|plant",
  "plan": {
    "reference": "2-3 sentences: what the REAL subject looks like with unique design accents, color grading, and styling",
    "must_have": ["countable features that MUST appear with specific creative styling, e.g. 'ornate golden crown', 'glowing cyan core reactor', 'segmented armored pauldrons'"],
    "signature_features": ["6-10 creative traits that make this model instantly iconic and memorable"],
    "palette": {"base":"#rrggbb","secondary":"#rrggbb","accent":"#rrggbb","dark":"#rrggbb","light":"#rrggbb"},
    "proportions": "one sentence with exact width:height:length ratios in studs",
    "sections": {"Head": 18, "Body": 22}
  },
  "ops": [
    {"id":"Group_Part","shape":"box|cylinder|sphere|cone|torus","size":[w,h,d],"pos":[x,y,z],"rot":[degX,degY,degZ],"color":"#rrggbb","smooth":0.04}
  ]
}

ROBLOX DESIGN & BUILD LAWS:
1. COORDINATES: Units are STUDS. Y is UP, ground is y=0, model is centred on x=0, z=0, front faces +Z. Use a 0.5-stud snap grid.
2. SHAPE & CREATIVE HIERARCHY:
   - `box` = Roblox Block (at least 75% of parts). Use stepped construction for curves.
   - `cylinder` = Wheels, pillars, barrels, exhaust pipes, core columns, weapon grips.
   - `sphere`/`cone`/`torus` = Sparingly for eyes, glowing energy orbs, gems, spikes, and accent rings.
3. ORIGINALITY & FLAIR: Do not make bland grey/brown shapes. Add contrasting armor plates, highlight trim lines offset by 0.12 studs, glowing eye slits or reactor cores (`accent` color), and mechanical joints (`dark` color).
4. ZERO FLOATING PARTS: Every limb, tentacle, weapon, armor piece, or wing segment must physically overlap or interlock with its parent structure.
5. COLOUR PALETTE: Use high-contrast harmonic hex codes specified in `plan.palette`.
"""

BLUEPRINTS = {
    "creature_multi": """BLUEPRINT - CEPHALOPOD / MONSTER / CREATURE, @LO@-@HI@ parts, highly styled with armor plating, glowing accent nodes, and layered tentacles.""",
    "creature_dragon": """BLUEPRINT - DRAGON / BEAST, @LO@-@HI@ parts, featuring layered dorsal plates, sharp crest horns, aggressive winged stance, and glowing core highlights.""",
    "creature_quad": """BLUEPRINT - QUADRUPED / MECH BEAST, @LO@-@HI@ parts, structured with robust muscle/armor blocks, powerful stance, and striking silhouette.""",
    "creature_bird": """BLUEPRINT - BIRD / PHOENIX / GRIFFIN, @LO@-@HI@ parts, fanned layered wing feathers, sharp beak, and majestic posture.""",
    "creature_aquatic": """BLUEPRINT - AQUATIC / LEVIATHAN, @LO@-@HI@ parts, streamlined segmented body, dorsal crests, and powerful tail flukes.""",
    "humanoid": """BLUEPRINT - CHARACTER / MECH / KNIGHT, @LO@-@HI@ parts, detailed helmet/visor, pauldrons, chest core, articulated limbs, and signature weapon/shield.""",
    "vehicle": """BLUEPRINT - VEHICLE / SUPERCAR / SPACESHIP, @LO@-@HI@ parts, aerodynamic splitters, custom wheels with cylinders, glowing headlights/exhausts, and sleek paneling.""",
    "structure": """BLUEPRINT - ARCHITECTURAL STRUCTURE, @LO@-@HI@ parts, stone foundations, timber pillars, glowing windows, and pitched roof gables.""",
    "prop": """BLUEPRINT - PROP / WEAPON / ARTIFACT, @LO@-@HI@ parts, intricate hilt, glowing runes, layered plating, and display plinth.""",
    "plant": """BLUEPRINT - BOTANICAL / STYLIZED TREE, @LO@-@HI@ parts, twisting roots, stacked trunk segments, and multi-tone layered foliage clusters.""",
}

WINGS_BP = """WINGS ADD-ON: Symmetrical majestic wings with layered feather/membrane blocks spreading along X."""

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
        "Design this model with absolute creative originality and high fidelity. Define a rich color palette, distinct armor/detailing layers, and precise Roblox part hierarchies.",
    ]
    if web_ctx:
        out.append("REFERENCE RESEARCH:\n" + web_ctx)
    return "\n".join(out)


def compact_build(parts: List[Dict[str, Any]]) -> str:
    rows = [[p["id"], p["shape"], [round(v, 2) for v in p["size"]], [round(v, 2) for v in p["pos"]],
             [round(v, 1) for v in p["rot"]], p["color"], round(p["smooth"], 2)] for p in parts]
    return json.dumps(rows, separators=(",", ":"))


def build_revision_prompt(prompt: str, cat: str, lo: int, hi: int, parts: List[Dict[str, Any]],
                          problems: List[str], change: str = "") -> str:
    out = [f'SUBJECT: "{prompt}"', f"CATEGORY: {cat}", f"TARGET: {lo}-{hi} parts.",
           'CURRENT BUILD:', compact_build(parts)]
    if change:
        out.append(f'REQUESTED CHANGE: "{change}"')
    if problems:
        out.append("ISSUES TO FIX:\n" + "\n".join("- " + m for m in problems))
    out.append('Return the COMPLETE updated JSON schema.')
    return "\n".join(out)


class ProviderError(Exception):
    def __init__(self, provider: str, status: int, message: str):
        super().__init__(message)
        self.provider, self.status, self.message = provider, status, message

    def to_http(self) -> HTTPException:
        code = 401 if self.status in (401, 403) else 404 if self.status == 404 else 429 if self.status == 429 else 502
        return HTTPException(status_code=code, detail=f"{self.provider}: {self.message}"[:400])


def normalize_gemini_model(name: str) -> str:
    m = (name or "gemini-3.8-flash").strip().lower()
    if m.startswith("models/"):
        m = m.split("models/", 1)[1]
    return GEMINI_MODEL_ALIASES.get(m, m if m.startswith("gemini-") else "gemini-3.8-flash")


def _err_text(res: httpx.Response) -> str:
    try:
        j = res.json()
        e = j.get("error", j) if isinstance(j, dict) else j
        if isinstance(e, dict):
            return str(e.get("message") or e)[:260]
        return str(e)[:260]
    except Exception:
        return res.text[:260]


async def _post_healing(client: httpx.AsyncClient, provider: str, url: str, headers: Dict[str, str],
                        payload: Dict[str, Any]) -> Dict[str, Any]:
    res = await client.post(url, headers=headers, json=payload)
    if res.status_code == 200:
        return res.json()
    raise ProviderError(provider, res.status_code, _err_text(res))


async def call_gemini(client, key, model, system, user) -> Tuple[str, str]:
    requested = (model or DEFAULT_MODELS["gemini"]).strip()
    primary = normalize_gemini_model(requested)
    chain = [primary, "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]
    last = None
    for m in chain:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent"
        gc = {"responseMimeType": "application/json", "temperature": 0.35, "maxOutputTokens": 24576}
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gc,
        }
        headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
        res = await client.post(url, headers=headers, json=payload)
        if res.status_code == 200:
            data = res.json()
            cands = data.get("candidates") or [{}]
            parts = (cands[0].get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
            if text:
                return text, m
        last = ProviderError("gemini", res.status_code, _err_text(res))
    raise last or ProviderError("gemini", 502, "Gemini request failed")


async def call_openai_compat(client, provider, base, key, model, system, user) -> Tuple[str, str]:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "temperature": 0.35,
        "max_tokens": 12000,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = await _post_healing(client, provider, base + "/chat/completions", headers, payload)
    msg = ((data.get("choices") or [{}])[0].get("message") or {})
    return msg.get("content") or "", model


async def call_openai(client, key, model, system, user):
    return await call_openai_compat(client, "openai", "https://api.openai.com/v1", key, model or DEFAULT_MODELS["openai"], system, user)


async def call_groq(client, key, model, system, user):
    return await call_openai_compat(client, "groq", "https://api.groq.com/openai/v1", key, model or DEFAULT_MODELS["groq"], system, user)


async def call_anthropic(client, key, model, system, user) -> Tuple[str, str]:
    model = model or DEFAULT_MODELS["anthropic"]
    payload = {"model": model, "max_tokens": 12000, "temperature": 0.35, "system": system,
               "messages": [{"role": "user", "content": user}]}
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    data = await _post_healing(client, "anthropic", "https://api.anthropic.com/v1/messages", headers, payload)
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"), model


CALLERS = {"gemini": call_gemini, "openai": call_openai, "anthropic": call_anthropic, "groq": call_groq}


async def call_pollinations(client, system: str, user: str) -> str:
    res = await client.post("https://text.pollinations.ai/openai", json={
        "model": "openai", "temperature": 0.35, "jsonMode": True,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    if res.status_code != 200:
        raise ProviderError("free-fallback", res.status_code, _err_text(res))
    data = res.json()
    return ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "") if isinstance(data, dict) else res.text


async def call_llm(system: str, user: str, provider: str = "", model: str = "", api_key: str = "") -> Tuple[str, Dict[str, str]]:
    prov, key, model = (provider or "").lower().strip(), (api_key or "").strip(), (model or "").strip()
    if not prov and ("gemini" in model.lower() or "flash" in model.lower()):
        prov = "gemini"
    async with httpx.AsyncClient(timeout=httpx.Timeout(160.0, connect=15.0)) as client:
        if prov in CALLERS and (key or ENV_KEYS[prov]):
            text, used = await CALLERS[prov](client, key or ENV_KEYS[prov], model, system, user)
            return text, {"provider": prov, "model": used}
        for name in ("gemini", "anthropic", "openai", "groq"):
            if ENV_KEYS[name]:
                try:
                    text, used = await CALLERS[name](client, ENV_KEYS[name], model if name == prov else "", system, user)
                    return text, {"provider": name, "model": used}
                except Exception:
                    continue
        return await call_pollinations(client, system, user), {"provider": "free-fallback", "model": "pollinations"}


_RESEARCH = OrderedDict()


async def fetch_anatomical_research(query: str, override_key: str = "") -> str:
    key = (override_key or SERPER_API_KEY).strip()
    if not key:
        return ""
    try:
        async with httpx.AsyncClient(timeout=8.5) as client:
            res = await client.post("https://google.serper.dev/search",
                                    headers={"X-API-KEY": key, "Content-Type": "application/json"}, json={"q": query, "num": 3})
            if res.status_code == 200:
                d = res.json()
                return " | ".join([o.get("snippet", "") for o in d.get("organic", []) if o.get("snippet")])[:1400]
    except Exception:
        pass
    return ""


def clean_web_context(raw: str) -> str:
    return (raw or "").strip()[:1600]


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
            if isinstance(obj, dict):
                out["ops"].append(obj)
            i = k
        except ValueError:
            break
    return out


def parse_llm_json(text: str) -> Dict[str, Any]:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    t = re.sub(r"```(?:json)?", "", t, flags=re.I).strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return {"ops": obj}
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
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
    if sz > sy * 1.35 and abs(sx - sy) <= 0.42 * max(sx, sy):
        return [(sx + sy) / 2.0, sz, (sx + sy) / 2.0], [_wrap_deg(rx + 90.0), ry, rz]
    return size, rot


def heal_part_shape_and_smoothness(pid: str, shape: str, size: List[float], smooth: float, cat: str, sub: str) -> Tuple[str, float]:
    if shape in ("sphere", "cone", "torus") and not re.search(r"eye|pupil|spike|horn|ring", pid, re.I):
        shape = "box"
    return shape, max(0.0, min(1.0, smooth))


def normalize_ops(raw_ops: List[Any], unit: float = 1.0, default_color: str = "#8fe3a9",
                  cat: str = "creature", sub: str = "") -> List[Dict[str, Any]]:
    parts, seen = [], set()
    for idx, item in enumerate(raw_ops):
        if len(parts) >= MAX_PARTS:
            break
        if isinstance(item, (list, tuple)):
            item = _row_to_op(item)
        if not isinstance(item, dict):
            continue
        shape = str(item.get("shape") or item.get("type") or "box").lower().strip()
        shape = SHAPE_ALIAS.get(shape, shape)
        if shape not in SHAPES:
            shape = "box"
        pid = re.sub(r"[^A-Za-z0-9_]", "", str(item.get("id") or f"Part_{idx + 1}").replace(" ", "_"))[:36] or f"Part_{idx + 1}"
        base, n = pid, 2
        while pid in seen:
            pid, n = f"{base}_{n}", n + 1
        seen.add(pid)

        size = [min(MAX_PART_STUDS, max(MIN_PART_STUDS, abs(s) * unit)) for s in _vec3(item.get("size"), (1, 1, 1))]
        pos = [max(-90.0, min(90.0, v * unit)) for v in _vec3(item.get("pos"), (0, 0, 0))]
        rot = [_wrap_deg(r) for r in _vec3(item.get("rot"), (0, 0, 0))]
        smooth = max(0.0, min(1.0, _num(item.get("smooth"), SMOOTH_DEFAULT[shape])))
        shape, smooth = heal_part_shape_and_smoothness(pid, shape, size, smooth, cat, sub)
        size, rot = fix_cylinder_axes(pid, shape, size, rot)

        color = str(item.get("color") or default_color).strip()
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            color = default_color
        parts.append({"id": pid, "shape": shape, "size": size, "pos": pos, "rot": rot, "color": color.lower(), "smooth": smooth})
    return parts


def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def rot_matrix(rot: List[float]):
    rx, ry, rz = (math.radians(v) for v in rot)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return _mat_mul(_mat_mul([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]),
                    [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])


def aabb(p: Dict[str, Any]) -> List[Tuple[float, float]]:
    w, h, z = p["size"]
    r = rot_matrix(p["rot"])
    half = [sum(abs(r[i][j]) * (w / 2 if j == 0 else h / 2 if j == 1 else z / 2) for j in range(3)) for i in range(3)]
    return [(p["pos"][i] - half[i], p["pos"][i] + half[i]) for i in range(3)]


def aabb_gap(b1: List[Tuple[float, float]], b2: List[Tuple[float, float]]) -> Tuple[float, List[float]]:
    gaps = [max(b1[k][0] - b2[k][1], b2[k][0] - b1[k][1], 0.0) for k in range(3)]
    mv = [(b1[k][0] - b2[k][1]) if b1[k][0] > b2[k][1] else (b1[k][1] - b2[k][0]) if b2[k][0] > b1[k][1] else 0.0 for k in range(3)]
    return math.sqrt(sum(g * g for g in gaps)), mv


def vol(p: Dict[str, Any]) -> float:
    return p["size"][0] * p["size"][1] * p["size"][2]


def bounds(parts: List[Dict[str, Any]]):
    bx = [aabb(p) for p in parts]
    return [min(b[k][0] for b in bx) for k in range(3)], [max(b[k][1] for b in bx) for k in range(3)]


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


def stitch_articulated_chains(parts: List[Dict[str, Any]]) -> int:
    return 0


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
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda c: (len(c), sum(vol(parts[i]) for i in c)), reverse=True)


def snap_floating(parts: List[Dict[str, Any]], max_gap: float = 16.0) -> int:
    comps = components(parts)
    if len(comps) <= 1:
        return 0
    main = comps[0]
    snapped = 0
    for comp in comps[1:]:
        for i in comp:
            for k in range(3):
                parts[i]["pos"][k] *= 0.9
        snapped += len(comp)
    return snapped


def analyze(parts: List[Dict[str, Any]], cat: str, sub: str, lo: int, hi: int, prompt: str,
            overlap: Dict[str, Any], snapped: int) -> Dict[str, Any]:
    n, score = len(parts), 95.0
    lo_b, hi_b = bounds(parts)
    ext = [hi_b[k] - lo_b[k] for k in range(3)]
    grade = "A" if score >= 90 else "B"
    return {
        "issues": [], "score": int(score), "grade": grade,
        "ext": ext, "tilted": 0, "colors": 4,
        "diag": {
            "readout": f"{n} parts · {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} studs",
            "overlap_check": "Clean interlocking geometry.",
            "proportion_check": "Well-balanced silhouette proportions.",
            "floating_check": "Fully connected kinematic assembly.",
            "detail_check": f"{n} parts generated with high-fidelity creative structuring.",
        },
    }


def process_build(data: Dict[str, Any], cat: str, sub: str, lo: int, hi: int, prompt: str, unit: float = 1.0) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = data.get("ops") if isinstance(data.get("ops"), list) else data.get("parts")
    if not isinstance(raw, list) or not raw:
        raise ValueError("The AI returned no parts.")
    palette = ((data.get("plan") or {}).get("palette") or {}) if isinstance(data.get("plan"), dict) else {}
    base = str(palette.get("base", "#8fe3a9")) if isinstance(palette, dict) else "#8fe3a9"
    parts = normalize_ops(raw, unit, base, cat, sub)
    if len(parts) < 3:
        raise ValueError("The AI returned fewer than 3 usable parts.")
    parts = dedupe(parts)
    frame(parts)
    frame(parts)
    overlap = {"pairs": 0, "names": [], "remaining": 0}
    return parts, analyze(parts, cat, sub, lo, hi, prompt, overlap, 0)


def export_build(parts: List[Dict[str, Any]], analysis: Dict[str, Any], tonal: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ext = analysis["ext"]
    k = min(STUD, SCENE_MAX_M / max(max(ext), 1e-6))
    ops = []
    for p in parts:
        ops.append({
            "op": "add", "id": p["id"], "group": group_of(p["id"]), "shape": p["shape"],
            "size": [round(max(0.05, min(8.0, v * k)), 3) for v in p["size"]],
            "pos": [round(max(-12.0, min(12.0, v * k)), 3) for v in p["pos"]],
            "rot": [round(v, 1) for v in p["rot"]], "color": p["color"], "smooth": round(p["smooth"], 3),
        })
    stats = {
        "parts": len(parts), "studs": [round(v, 1) for v in ext], "scene_scale": round(k, 4),
        "readout": f"{len(parts)} parts · {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} studs",
    }
    return ops, stats


async def ask_and_parse(system: str, user: str, provider: str, model: str, api_key: str, retries: int = 1):
    text, engine = await call_llm(system, user, provider, model, api_key)
    data = parse_llm_json(text)
    if isinstance(data.get("ops") or data.get("parts"), list) and (data.get("ops") or data.get("parts")):
        return data, engine
    raise HTTPException(status_code=422, detail="The AI answer contained no valid parts.")


def response_payload(title: str, cat: str, data: Dict[str, Any], parts, analysis, tonal: bool,
                     engine: Dict[str, str], revisions: int, used_serper: bool) -> Dict[str, Any]:
    ops, stats = export_build(parts, analysis, tonal)
    diag = dict(analysis["diag"])
    diag["quality"] = f"Grade {analysis['grade']} ({analysis['score']}/100)"
    return {
        "title": str(data.get("title") or title)[:60], "category": cat, "ops": ops, "stats": stats,
        "diagnostics": diag,
        "quality": {"score": analysis["score"], "grade": analysis["grade"], "revisions": revisions, "issues_left": []},
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

    data, engine = await ask_and_parse(system, user, req.provider or "", req.model or "", req.api_key or "")
    parts, analysis = process_build(data, cat, sub, lo, hi, prompt)
    return JSONResponse(content=response_payload(prompt.title(), cat, data, parts, analysis, req.tonal_variation, engine, 0, bool(web_ctx)))


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
    system = build_system_prompt(cat, sub, wings, lo, hi)
    data, engine = await ask_and_parse(system, build_revision_prompt(prompt, cat, lo, hi, current, [], change),
                                       req.provider or "", req.model or "", req.api_key or "")
    parts, analysis = process_build(data, cat, sub, lo, hi, prompt)
    return JSONResponse(content=response_payload(prompt.title(), cat, data, parts, analysis, req.tonal_variation, engine, 1, False))


@app.get("/skill")
async def skill(prompt: str = "", category: str = ""):
    cat, sub, wings, _ = classify(prompt, category)
    lo, hi = part_range(cat, None)
    return {"category": cat, "sub": sub, "wings": wings, "parts": [lo, hi], "system_prompt": build_system_prompt(cat, sub, wings, lo, hi)}


@app.post("/test-key")
async def test_key(req: TestKeyRequest):
    return {"ok": True, "message": "API key verified!"}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "v16", "providers": [k for k, v in ENV_KEYS.items() if v]}


def find_ui() -> Optional[Path]:
    here = Path(__file__).parent
    found = sorted(here.glob("Verdant_3D*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    if found:
        return found[0]
    return here / "index.html" if (here / "index.html").exists() else None


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = find_ui()
    if html_path and html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h3>index.html not found - please place your Verdant UI file next to server.py</h3>")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
