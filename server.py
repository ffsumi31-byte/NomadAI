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

app = FastAPI(title="Verdant 3D — Production Ready Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "gemini-3.8-flash")

MASTER_SCULPTING_SKILL = """You are a Roblox Game Asset Builder. 
You design clean, iconic, blocky 3D models tailored specifically for modern Roblox games (sharp hard-cornered geometry, 30 to 55 parts max, vibrant and bold color palette, zero micro-clutter). Reply with ONLY valid JSON (no markdown fences):
{
  "title": "Roblox Style Model",
  "category": "creature|humanoid|vehicle|structure|prop",
  "ops": [
    {
      "op": "add",
      "id": "PartName",
      "shape": "box|sphere|cylinder|cone|torus",
      "size": [width_x, height_y, depth_z],
      "pos": [x, y, z],
      "rot": [degX, degY, degZ],
      "color": "#rrggbb",
      "smooth": 0.02
    }
  ]
}
ROBLOX GAME-READY LAWS:
1. PART BUDGET: 30 to 55 parts max. Clean, readable silhouettes.
2. HARD CORNERS: Use "box" shapes with sharp edges (smooth: 0.01 to 0.03).
3. BOLD PALETTE: Use 3 to 4 flat, vibrant, saturated colors. No muddy textures.
4. COORDINATES: Meters, Y is up, ground is y=0. Ensure stability and zero Z-fighting."""


class GenerateRequest(BaseModel):
    prompt: str
    web_context: Optional[str] = ""
    use_serper: bool = True


async def fetch_anatomical_research(query: str) -> str:
    async with httpx.AsyncClient(timeout=10.0) as client:
        if SERPER_API_KEY:
            try:
                res = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
                    json={"q": f"{query} geometry layout block structure colors", "num": 3},
                )
                if res.status_code == 200:
                    data = res.json()
                    bits = []
                    for org in data.get("organic", [])[:3]:
                        if s := org.get("snippet"):
                            bits.append(s)
                    if bits:
                        return " | ".join(bits)[:600]
            except Exception:
                pass
    return ""


async def call_llm_architect(system_prompt: str, user_prompt: str) -> str:
    async with httpx.AsyncClient(timeout=95.0) as client:
        # 1. Google Gemini (Cascading through Gemini 3.8+ stable tiers to bypass 503 spikes)
        if GEMINI_API_KEY:
            models_to_try = [AI_MODEL, "gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-flash"]
            models_to_try = [m for i, m in enumerate(models_to_try) if m and m not in models_to_try[:i]]

            for model in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
                try:
                    res = await client.post(
                        url,
                        json={
                            "systemInstruction": {"parts": [{"text": system_prompt}]},
                            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                            "generationConfig": {
                                "responseMimeType": "application/json",
                                "temperature": 0.3,
                                "maxOutputTokens": 8192,
                            },
                        },
                    )
                    if res.status_code == 200:
                        data = res.json()
                        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                        return "".join(p.get("text", "") for p in parts)
                    elif res.status_code == 503:
                        continue  # Try next stable fallback tier on high demand
                except Exception:
                    continue

        # 2. Backup Cloud / Pollinations
        res = await client.post(
            "https://text.pollinations.ai/openai",
            json={
                "model": "openai",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "jsonMode": True,
            },
        )
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, dict) and "ops" in data:
                return json.dumps(data)
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    raise HTTPException(status_code=502, detail="All AI providers are currently experiencing heavy traffic. Please try again.")


def parse_and_validate_build(raw_text: str, prompt: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("AI output did not contain valid JSON.")

    data = json.loads(cleaned[start : end + 1])
    raw_ops = data.get("ops") or data.get("parts") or []
    if not isinstance(raw_ops, list) or not raw_ops:
        raise ValueError("AI JSON contained no parts.")

    ops = []
    for idx, item in enumerate(raw_ops[:60]):
        if not isinstance(item, dict):
            continue
        shape = str(item.get("shape") or item.get("s") or "box").lower().strip()
        if shape not in {"box", "sphere", "cylinder", "cone", "torus"}:
            shape = "box"

        rs = item.get("size") or item.get("sz") or [0.5, 0.5, 0.5]
        size = [round(max(0.04, min(8.0, float(rs[i] if i < len(rs) else 0.5))), 3) for i in range(3)]

        rp = item.get("pos") or item.get("po") or [0, size[1] / 2, 0]
        pos = [round(max(-10.0, min(10.0, float(rp[i] if i < len(rp) else 0.0))), 3) for i in range(3)]

        rr = item.get("rot") or item.get("ro") or [0, 0, 0]
        rot = [round(max(-360.0, min(360.0, float(rr[i] if i < len(rr) else 0.0))), 1) for i in range(3)]

        color = str(item.get("color") or item.get("c") or "#8fe3a9").strip()
        if not re.match(r"^#[0-9a-fA-F]{6}$", color):
            color = "#8fe3a9"

        ops.append({
            "op": "add",
            "id": str(item.get("id") or item.get("n") or f"Part_{idx+1}")[:40],
            "shape": shape,
            "size": size,
            "pos": pos,
            "rot": rot,
            "color": color,
            "smooth": 0.02,
        })

    # Grounding check
    if ops:
        min_y = min(p["pos"][1] - p["size"][1] / 2 for p in ops)
        if abs(min_y) > 0.01:
            offset_y = -min_y
            for p in ops:
                p["pos"][1] = round(p["pos"][1] + offset_y, 3)

    return {
        "title": str(data.get("title") or prompt.title()),
        "ops": ops,
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    web_ctx = req.web_context.strip() if req.web_context else ""
    if not web_ctx and req.use_serper:
        web_ctx = await fetch_anatomical_research(prompt)

    user_instruction = f"Subject: '{prompt}'. Build a clean, game-ready Roblox asset using 30 to 55 crisp parts."
    try:
        raw_output = await call_llm_architect(MASTER_SCULPTING_SKILL, user_instruction)
        result = parse_and_validate_build(raw_output, prompt)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return HTMLResponse("<h3>index.html not found next to server.py</h3>")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
