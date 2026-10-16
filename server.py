import os
import re
import json
import math
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Verdant 3D — Master Block-Sculpting Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API Keys (Set in Render / Koyeb Environment Variables or PowerShell) ───
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "")

# ─── THE MASTER DESIGN SKILL: Pure-Intelligence Block & Brick Sculpting Curriculum ───
MASTER_SCULPTING_SKILL = """You are a Master 3D Block-Sculpting Architect (Roblox Studio / Lego Master Builder tier).
You have ZERO pre-made assets. You synthesize complex, recognizable 75-to-135 part 3D models purely through spatial mathematics, kinematic chains, and stepped block layering.

═══════════════════════════════════════════════════════════════════════════════
1. CORE DESIGN PHILOSOPHY (CRISP ARTICULATED BLOCK SCULPTING)
═══════════════════════════════════════════════════════════════════════════════
- NEVER represent an animal, character, or vehicle with 5-10 giant blobs.
- Target **75 to 130 distinct parts** (`op: "add"`).
- Prefer crisp, architectural block/wedge geometry (`"box"` with `smooth: 0.04` to `0.12`, `"cylinder"`, `"cone"`). Avoid giant mushy spheres; build heads, snouts, chests, and limbs out of faceted, angled, multi-layered boxes and plates!
- TWO-TONE & THREE-TONE SHADING: Never paint a model one flat color. Every creature/vehicle needs:
  1. Primary body hex color (e.g., `#c92a2a` crimson)
  2. Lighter underbelly/chest/muzzle highlight hex (e.g., `#f3d9a4` cream/gold)
  3. Darker structural/joint/trim accent hex (e.g., `#7a1818` dark maroon or `#24292e` charcoal)
  4. High-contrast micro-details (sclera `#ffffff`, iris `#f59f00`, pupil `#111111`, claws/horns `#f8f9fa`).

═══════════════════════════════════════════════════════════════════════════════
2. FIVE MANDATORY SPATIAL ENGINEERING TECHNIQUES
═══════════════════════════════════════════════════════════════════════════════
A. KINEMATIC CHAINS (Necks, Tails, Tentacles, Limbs, Serpents):
   - Never use 1 straight bar for a neck, tail, or tentacle.
   - Construct curves out of 4 to 7 sequential segments (`Neck1`, `Neck2`, `Neck3`... or `Tentacle1_Seg1`..`Seg5`).
   - Each child segment steps along the curve vector using trigonometry:
     `next_pos = parent_pos + rotated_offset`, progressively tapering `size` (e.g., `[0.6, 0.7, 0.6]` -> `[0.5, 0.65, 0.5]` -> `[0.38, 0.55, 0.38]`) and incrementing pitch/yaw rotation (`rot: [15, 0, 0]`, `[32, 0, 0]`, `[48, 0, 0]`).

B. RADIAL FEATHER / MANE / ARMOR LAYERING (Wings, Manes, Crests, Spoilers):
   - For wings (like a Phoenix or Dragon): Build a 3-segment articulated wing bone (`WingShoulder_L`, `WingForearm_L`, `WingTipBone_L`) angled outward and upward. Attach 6 to 10 individual flat feather plates (`size: [0.18, 1.1, 0.35]`) fanned out along the wing bones with progressive Z/X rotations (`rot: [12, -15, 25]`, `[12, -15, 38]`, `[12, -15, 52]`) and alternating primary/gold tip colors.
   - For manes/hair/fur (like a Lion or Fighter): Frame the head with 8 to 14 stepped, angled plates around the crown, cheeks, and chin to create a rich 3D silhouette.

C. MICRO-DETAILED FACIAL STACKING (Eyes, Snouts, Beaks, Mouths):
   - A head is NEVER just 1 box. Build:
     `Cranium` + `BrowRidge` + `SnoutUpper` + `JawLower` + `NoseTip` + `Ear_L`/`Ear_R` + `EyeWhite_L`/`EyeWhite_R` + `EyePupil_L`/`EyePupil_R` (offset `0.02m` forward in Z so they never flicker/Z-fight with the face!).

D. ARTICULATED LIMBS, HANDS, PAWS & CLAWS:
   - Legs/Arms must have: `UpperLimb` (thigh/bicep) + `LowerLimb` (shin/forearm) + `Foot/Paw/Fist` + 2 to 3 individual `Claw` or `Toe` parts resting flat on `y = height / 2`.
   - Dynamic Pose: Angle limbs slightly (`rot: [12, 0, 0]` vs `[-10, 0, 0]`) so the model looks alive and mid-action rather than stiff.

E. VEHICLES & ARCHITECTURE (Wheel Arches, Trim, Recessed Windows):
   - Vehicles must abide by the **2.5-to-3.2 Wheel Height Rule**: If wheels have diameter `D` (e.g., `0.85m`), total car height MUST be ~`2.2m` to `2.6m` (`2.6 * D`), length ~`5.2m`, width ~`2.4m`.
   - Build separate `ChassisBase`, `FrontSplitter`, `Hood`, `Headlight_L/R`, `Grille`, `Windshield` (angled `rot: [-28, 0, 0]`), `A_Pillar_L/R`, `Roof`, `SideDoor_L/R`, `RearQuarter_L/R`, `SpoilerWing`, `SpoilerStrut_L/R`, `Taillight_L/R`, `WheelTire_FL/FR/RL/RR`, and `WheelRim_FL/FR/RL/RR`.

═══════════════════════════════════════════════════════════════════════════════
3. ANTI-FLICKER (COPLANAR Z-FIGHTING) & GROUNDING LAWS
═══════════════════════════════════════════════════════════════════════════════
- COORDINATES: Meters (`1m ≈ 8 Roblox studs`). `Y` is UP (`+Y`), `X` is Left/Right (`-X` left, `+X` right), `Z` is Front/Back (`+Z` front, `-Z` back). Ground is `y = 0`.
- Whole model fits within a `6.0m x 6.0m x 6.0m` stage (`x, z` in `[-3.0, 3.0]`, `y` in `[0.0, 6.0]`).
- NO COPLANAR Z-FIGHTING: When two parts overlap (like a racing stripe on a hood, an eye on a face, or cabin glass inside a pillar), NEVER give them the exact same X, Y, or Z boundary (`pos ± size/2`). Always inset or outset the detail part by at least `0.02m` to `0.04m` so surfaces never flicker!

═══════════════════════════════════════════════════════════════════════════════
4. OUTPUT SCHEMA (STRICT RAW JSON ONLY)
═══════════════════════════════════════════════════════════════════════════════
Return ONLY valid JSON with no markdown fences:
{
  "title": "Model Name",
  "category": "creature|humanoid|vehicle|structure|prop",
  "ops": [
    {
      "op": "add",
      "id": "DescriptivePartName",
      "shape": "box|sphere|cylinder|cone|torus",
      "size": [width_x, height_y, depth_z],
      "pos": [center_x, center_y, center_z],
      "rot": [deg_x, deg_y, deg_z],
      "color": "#rrggbb",
      "smooth": 0.06
    }
  ]
}"""


class GenerateRequest(BaseModel):
    prompt: str
    web_context: Optional[str] = ""
    use_serper: bool = True
    current_ops: Optional[List[Dict[str, Any]]] = None
    revision_note: Optional[str] = ""


# ─── 1. Live Web Anatomical & Proportional Research ───
async def fetch_anatomical_research(query: str) -> str:
    async with httpx.AsyncClient(timeout=10.0) as client:
        if SERPER_API_KEY:
            try:
                res = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                    json={
                        "q": f"{query} anatomy proportions body parts segments colors visual breakdown",
                        "num": 5,
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    bits = []
                    if kg := data.get("knowledgeGraph"):
                        bits.append(f"{kg.get('title', '')}: {kg.get('description', '')}")
                    for org in data.get("organic", [])[:4]:
                        if s := org.get("snippet"):
                            bits.append(s)
                    if bits:
                        return " | ".join(bits)[:900]
            except Exception:
                pass

        try:
            words = [
                w for w in re.sub(r"[^a-z0-9 ]", "", query.lower()).split()
                if len(w) > 2 and w not in {"make", "build", "create", "the", "with", "from"}
            ]
            subj = words[-1] if words else query
            res = await client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{subj}")
            if res.status_code == 200:
                extract = res.json().get("extract", "")
                if extract:
                    return extract[:500]
        except Exception:
            pass
    return ""


# ─── 2. Automated Coplanar Overlap (Z-Fighting) & Proportion Diagnostic Engine ───
def boxes_intersect_3d(a_min, a_max, b_min, b_max, eps: float = 0.005) -> bool:
    return (
        (a_min[0] < b_max[0] - eps and a_max[0] > b_min[0] + eps)
        and (a_min[1] < b_max[1] - eps and a_max[1] > b_min[1] + eps)
        and (a_min[2] < b_max[2] - eps and a_max[2] > b_min[2] + eps)
    )


def audit_and_fix_geometry(ops: List[Dict[str, Any]], category: str, prompt: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Performs the exact Distinctus-style Overlap Check and Proportion Check:
    1. Detects pairs of overlapping parts that share an identical boundary face (causing Studio stripe flickering)
       and micro-nudges the smaller detail part by 0.015m so it renders cleanly.
    2. Computes stud dimensions (1m = 8 studs) and checks vehicle wheel-to-height ratios & creature aspect ratios.
    """
    overlapping_pairs = []

    # Compute axis-aligned bounds for unrotated/lightly-rotated parts and fix coplanar Z-fighting
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            p1, p2 = ops[i], ops[j]
            s1, c1 = p1["size"], p1["pos"]
            s2, c2 = p2["size"], p2["pos"]

            min1 = [c1[k] - s1[k] / 2.0 for k in range(3)]
            max1 = [c1[k] + s1[k] / 2.0 for k in range(3)]
            min2 = [c2[k] - s2[k] / 2.0 for k in range(3)]
            max2 = [c2[k] + s2[k] / 2.0 for k in range(3)]

            if boxes_intersect_3d(min1, max1, min2, max2):
                # Check if they share any coplanar boundary face within 2mm (0.002m)
                shared_axis = -1
                for axis in range(3):
                    if abs(max1[axis] - max2[axis]) < 0.003 or abs(min1[axis] - min2[axis]) < 0.003:
                        shared_axis = axis
                        break
                if shared_axis != -1:
                    overlapping_pairs.append(f"{p1['id']} and {p2['id']}")
                    # Automatically resolve Z-fighting by nudging the smaller detail part outward by 0.012m
                    vol1 = s1[0] * s1[1] * s1[2]
                    vol2 = s2[0] * s2[1] * s2[2]
                    smaller = p2 if vol2 <= vol1 else p1
                    smaller["size"][shared_axis] = round(smaller["size"][shared_axis] + 0.016, 3)

    # Calculate overall bounding box in meters and Roblox Studs (1m = 8 studs)
    if ops:
        min_x = min(p["pos"][0] - p["size"][0] / 2 for p in ops)
        max_x = max(p["pos"][0] + p["size"][0] / 2 for p in ops)
        min_y = min(p["pos"][1] - p["size"][1] / 2 for p in ops)
        max_y = max(p["pos"][1] + p["size"][1] / 2 for p in ops)
        min_z = min(p["pos"][2] - p["size"][2] / 2 for p in ops)
        max_z = max(p["pos"][2] + p["size"][2] / 2 for p in ops)

        # Ground the entire model seamlessly at y = 0
        if abs(min_y) > 0.01:
            offset_y = -min_y
            for p in ops:
                p["pos"][1] = round(p["pos"][1] + offset_y, 3)
            max_y += offset_y
            min_y = 0.0

        width_studs = round((max_x - min_x) * 8.0, 1)
        height_studs = round((max_y - min_y) * 8.0, 1)
        depth_studs = round((max_z - min_z) * 8.0, 1)
    else:
        width_studs = height_studs = depth_studs = 0.0

    # Proportion check (e.g., checking vehicle height vs wheel diameter)
    proportion_warning = ""
    wheel_parts = [p for p in ops if any(w in p["id"].lower() for w in ("wheel", "tire"))]
    if wheel_parts and ("car" in prompt.lower() or "truck" in prompt.lower() or category == "vehicle"):
        avg_wheel_h = sum(max(p["size"]) for p in wheel_parts) / len(wheel_parts)
        wheel_studs = round(avg_wheel_h * 8.0, 1)
        ratio = round(height_studs / max(wheel_studs, 0.5), 1)
        if ratio > 3.6:
            proportion_warning = (
                f"Proportion check: it stands {height_studs} studs tall on {wheel_studs}-stud wheels, "
                f"about {ratio} wheels high. A car reads right at about 2.5–3.0."
            )
            # Auto-compress excessive roof height toward realistic 2.7x wheel ratio
            scale_y = (2.7 * wheel_studs) / height_studs
            for p in ops:
                if not any(w in p["id"].lower() for w in ("wheel", "tire", "rim")):
                    p["pos"][1] = round(p["pos"][1] * scale_y, 3)
                    p["size"][1] = round(max(0.05, p["size"][1] * scale_y), 3)
            height_studs = round(height_studs * scale_y, 1)

    overlap_note = ""
    if overlapping_pairs:
        sample = "; ".join(overlapping_pairs[:4])
        overlap_note = (
            f"Overlap check: {len(overlapping_pairs)} pairs of parts shared an identical surface "
            f"({sample}) — auto-offset by 0.1 studs to prevent surface flickering."
        )

    diagnostics = {
        "part_count": len(ops),
        "stud_dimensions": f"{width_studs} × {height_studs} × {depth_studs} studs",
        "overlap_check": overlap_note,
        "proportion_check": proportion_warning,
    }
    return ops, diagnostics


# ─── 3. Multi-Provider LLM Router (Supports High-Token 130-Part Output) ───
async def call_llm_architect(system_prompt: str, user_prompt: str) -> str:
    async with httpx.AsyncClient(timeout=95.0) as client:
        # 1. Anthropic Claude (Used by Distinctus for high-part-count spatial precision)
        if ANTHROPIC_API_KEY:
            model = AI_MODEL or "claude-sonnet-4-20250514"
            res = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 12000,
                    "temperature": 0.25,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            if res.status_code == 200:
                blocks = res.json().get("content", [])
                return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

        # 2. Google Gemini 2.5 Pro / Flash
        if GEMINI_API_KEY:
            model = AI_MODEL or "gemini-2.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            res = await client.post(
                url,
                json={
                    "systemInstruction": {"parts": [{"text": system_prompt}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.25,
                        "maxOutputTokens": 16384,
                    },
                },
            )
            if res.status_code == 200:
                candidates = res.json().get("candidates", [{}])
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts)

        # 3. OpenAI (GPT-4o / GPT-4.1)
        if OPENAI_API_KEY:
            model = AI_MODEL or "gpt-4o"
            res = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.25,
                    "max_completion_tokens": 12000,
                    "response_format": {"type": "json_object"},
                },
            )
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]

        # 4. Groq (Llama 3.3 70B / Qwen QwQ 32B)
        if GROQ_API_KEY:
            model = AI_MODEL or "llama-3.3-70b-versatile"
            res = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.25,
                    "max_tokens": 8192,
                    "response_format": {"type": "json_object"},
                },
            )
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]

        # 5. Built-in Keyless Cloud Fallback
        res = await client.post(
            "https://text.pollinations.ai/openai",
            json={
                "model": "openai-large",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.25,
                "jsonMode": True,
            },
        )
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, dict) and "ops" in data:
                return json.dumps(data)
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    raise HTTPException(status_code=502, detail="Could not reach any AI provider.")


# ─── 4. Parser & Normalizer (Supports up to 160 Parts) ───
VALID_SHAPES = {"box", "sphere", "cylinder", "cone", "torus"}
SHAPE_ALIASES = {
    "ball": "sphere", "ellipsoid": "sphere",
    "tube": "cylinder", "pillar": "cylinder", "rod": "cylinder",
    "pyramid": "cone", "roof": "cone", "wedge": "cone",
    "ring": "torus", "donut": "torus",
}


def clamp_f(val, lo: float, hi: float, d: float) -> float:
    try:
        v = float(val)
        return max(lo, min(hi, v)) if math.isfinite(v) else d
    except (TypeError, ValueError):
        return d


def parse_and_validate_build(raw_text: str, prompt: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```\$", "", raw_text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("AI output did not contain valid JSON.")

    data = json.loads(cleaned[start : end + 1])
    raw_ops = data.get("ops") or data.get("parts") or []
    if not isinstance(raw_ops, list) or not raw_ops:
        raise ValueError("AI JSON contained no parts.")

    category = str(data.get("category") or "creature").lower()
    ops = []

    for idx, item in enumerate(raw_ops[:160]):
        if not isinstance(item, dict):
            continue
        raw_shape = str(item.get("shape") or item.get("s") or "box").lower().strip()
        shape = SHAPE_ALIASES.get(raw_shape, raw_shape if raw_shape in VALID_SHAPES else "box")

        rs = item.get("size") or item.get("sz") or [0.5, 0.5, 0.5]
        size = [round(clamp_f(rs[i] if i < len(rs) else 0.5, 0.04, 8.0, 0.5), 3) for i in range(3)]

        rp = item.get("pos") or item.get("po") or [0, size[1] / 2, 0]
        pos = [round(clamp_f(rp[i] if i < len(rp) else 0.0, -10.0, 10.0, 0.0), 3) for i in range(3)]

        rr = item.get("rot") or item.get("ro") or [0, 0, 0]
        rot = [round(clamp_f(rr[i] if i < len(rr) else 0.0, -360.0, 360.0, 0.0), 1) for i in range(3)]

        color = str(item.get("color") or item.get("c") or "#8fe3a9").strip()
        if not re.match(r"^#[0-9a-fA-F]{6}\$", color):
            color = "#8fe3a9"

        # Default to crisp architectural brick edges (0.06) unless specified
        smooth = round(clamp_f(item.get("smooth", 0.06), 0.0, 1.0, 0.06), 2)
        part_id = str(item.get("id") or item.get("n") or f"Part_{idx+1}")[:40]

        ops.append({
            "op": "add",
            "id": part_id,
            "shape": shape,
            "size": size,
            "pos": pos,
            "rot": rot,
            "color": color,
            "smooth": smooth,
        })

    ops, diagnostics = audit_and_fix_geometry(ops, category, prompt)

    return {
        "title": f"{data.get('title') or prompt.title()} ({diagnostics['part_count']} parts · {diagnostics['stud_dimensions']})",
        "ops": ops,
        "diagnostics": diagnostics,
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    web_ctx = req.web_context.strip() if req.web_context else ""
    if not web_ctx and req.use_serper:
        web_ctx = await fetch_anatomical_research(prompt)

    user_instruction = (
        f"SUBJECT TO SCULPT: '{prompt}'\n"
        + (f"ANATOMICAL & VISUAL REFERENCE:\n{web_ctx}\n\n" if web_ctx else "\n")
        + "MANDATORY EXECUTION PLAN:\n"
        "1. Decompose the subject into 75 to 125 articulated, crisp 3D parts (`box` with `smooth: 0.05` to `0.12`, `cylinder`, `cone`).\n"
        "2. Build multi-segment kinematic chains for necks, tails, tentacles, or limbs.\n"
        "3. Build layered individual feather plates, mane blocks, fingers/claws, and multi-part facial features (brows, sclera, iris, pupil, jaw, snout).\n"
        "4. Use a rich 3-to-5 color palette (primary, lighter underbelly/highlight, darker structural shadow, eye/claw accents).\n"
        "5. Ensure zero floating parts and offset surface details by 0.02m so no two overlapping parts share an exact coplanar face."
    )

    if req.current_ops and req.revision_note:
        user_instruction += (
            f"\n\nEXISTING MODEL JSON TO REVISE:\n{json.dumps(req.current_ops)}\n"
            f"REVISION REQUEST: {req.revision_note}"
        )

    try:
        raw_output = await call_llm_architect(MASTER_SCULPTING_SKILL, user_instruction)
        result = parse_and_validate_build(raw_output, prompt)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "skill": "Master Articulated Block Sculptor (75-135 parts)"}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h3>Place your Verdant 3D <code>index.html</code> in the same directory as <code>server.py</code>.</h3>")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
