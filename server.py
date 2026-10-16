import os
import re
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Verdant 3D — Multi-AI + OpenRouter & Serper Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional Server-Side Environment Variables (Fallback if not passed from UI)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "gemini-2.5-flash")

# Maps non-existent/deprecated model aliases so Gemini API never rejects older 2.5 or placeholder strings
GEMINI_MODEL_ALIASES = {
    "gemini-3.8-flash": "gemini-2.5-flash",
    "gemini-3.6-flash": "gemini-2.5-flash",
    "gemini-2.5": "gemini-2.5-flash",
    "gemini-2.5-flash-latest": "gemini-2.5-flash",
    "gemini-2.5-pro-latest": "gemini-2.5-pro",
    "gemini-2.5-flash-preview-05-20": "gemini-2.5-flash",
    "gemini-2.5-pro-preview-05-06": "gemini-2.5-pro",
    "gemini-1.5-flash": "gemini-2.5-flash",
    "gemini-1.5-pro": "gemini-2.5-pro",
}

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
    provider: Optional[str] = ""
    model: Optional[str] = ""
    api_key: Optional[str] = ""
    serper_key: Optional[str] = ""


class TestKeyRequest(BaseModel):
    provider: str
    api_key: str
    model: Optional[str] = ""


def normalize_gemini_model(model_name: str) -> str:
    m = (model_name or "gemini-2.5-flash").strip()
    if m.startswith("models/"):
        m = m.split("models/", 1)[1]
    return GEMINI_MODEL_ALIASES.get(m, m)


async def fetch_anatomical_research(query: str, override_key: str = "") -> str:
    key = (override_key or SERPER_API_KEY).strip()
    if not key:
        return ""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            res = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": key, "Content-Type": "application/json"},
                json={"q": f"{query} visual anatomy proportions key parts colors", "num": 4},
            )
            if res.status_code == 200:
                data = res.json()
                bits = []
                if kg := data.get("knowledgeGraph"):
                    if desc := kg.get("description"):
                        bits.append(desc)
                for org in data.get("organic", [])[:4]:
                    if s := org.get("snippet"):
                        bits.append(s)
                if bits:
                    return " | ".join(bits)[:800]
        except Exception:
            pass
    return ""


async def call_gemini(client: httpx.AsyncClient, api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    primary = normalize_gemini_model(model)
    fallbacks = [
        primary,
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
    ]
    models_to_try = []
    for m in fallbacks:
        if m and m not in models_to_try:
            models_to_try.append(m)

    last_err = ""
    for m in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.3,
                "maxOutputTokens": 8192,
            },
        }
        try:
            res = await client.post(url, json=payload)
            if res.status_code == 200:
                data = res.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                if text:
                    return text
            elif res.status_code == 400 and "responseMimeType" in res.text:
                payload["generationConfig"].pop("responseMimeType", None)
                res2 = await client.post(url, json=payload)
                if res2.status_code == 200:
                    data = res2.json()
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts).strip()
                    if text:
                        return text
            last_err = f"Gemini ({m}) HTTP {res.status_code}: {res.text[:200]}"
            if res.status_code in (401, 403):
                break
        except Exception as e:
            last_err = str(e)
            continue

    raise RuntimeError(last_err or "Gemini request failed.")


async def call_openrouter(client: httpx.AsyncClient, api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    target_model = (model or "google/gemini-2.5-flash").strip()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://verdant3d.local",
        "X-Title": "Verdant 3D Builder",
    }
    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 8192,
    }
    res = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    if res.status_code != 200:
        raise RuntimeError(f"OpenRouter HTTP {res.status_code}: {res.text[:240]}")
    data = res.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


async def call_openai(client: httpx.AsyncClient, api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    target_model = (model or "gpt-4o").strip()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
    }
    res = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if res.status_code == 400 and "response_format" in res.text:
        payload.pop("response_format", None)
        res = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if res.status_code != 200:
        raise RuntimeError(f"OpenAI HTTP {res.status_code}: {res.text[:240]}")
    data = res.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


async def call_anthropic(client: httpx.AsyncClient, api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    target_model = (model or "claude-sonnet-4-20250514").strip()
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": target_model,
        "max_tokens": 8192,
        "temperature": 0.3,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    res = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
    if res.status_code != 200:
        raise RuntimeError(f"Anthropic HTTP {res.status_code}: {res.text[:240]}")
    data = res.json()
    blocks = data.get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


async def call_llm_architect(
    system_prompt: str,
    user_prompt: str,
    provider: str = "",
    model: str = "",
    api_key: str = "",
) -> str:
    prov = (provider or "").lower().strip()
    key = (api_key or "").strip()

    async with httpx.AsyncClient(timeout=95.0) as client:
        # 1. Explicit provider requested from UI
        if prov == "gemini" and (key or GEMINI_API_KEY):
            return await call_gemini(client, key or GEMINI_API_KEY, model or AI_MODEL, system_prompt, user_prompt)
        if prov == "openrouter" and (key or OPENROUTER_API_KEY):
            return await call_openrouter(client, key or OPENROUTER_API_KEY, model, system_prompt, user_prompt)
        if prov == "openai" and (key or OPENAI_API_KEY):
            return await call_openai(client, key or OPENAI_API_KEY, model, system_prompt, user_prompt)
        if prov == "anthropic" and (key or ANTHROPIC_API_KEY):
            return await call_anthropic(client, key or ANTHROPIC_API_KEY, model, system_prompt, user_prompt)

        # 2. Cascade through configured server environment keys
        errors: List[str] = []
        if GEMINI_API_KEY:
            try:
                return await call_gemini(client, GEMINI_API_KEY, model or AI_MODEL, system_prompt, user_prompt)
            except Exception as e:
                errors.append(str(e))

        if OPENROUTER_API_KEY:
            try:
                return await call_openrouter(client, OPENROUTER_API_KEY, model or "google/gemini-2.5-flash", system_prompt, user_prompt)
            except Exception as e:
                errors.append(str(e))

        if OPENAI_API_KEY:
            try:
                return await call_openai(client, OPENAI_API_KEY, model or "gpt-4o", system_prompt, user_prompt)
            except Exception as e:
                errors.append(str(e))

        if ANTHROPIC_API_KEY:
            try:
                return await call_anthropic(client, ANTHROPIC_API_KEY, model or "claude-sonnet-4-20250514", system_prompt, user_prompt)
            except Exception as e:
                errors.append(str(e))

        # 3. Free cloud fallback if no keys are configured
        try:
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
        except Exception as e:
            errors.append(str(e))

    detail_msg = " | ".join(errors) if errors else "No AI API key provided or all providers failed."
    raise HTTPException(status_code=502, detail=detail_msg)


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
            "smooth": round(max(0.0, min(1.0, float(item.get("smooth", 0.02)))), 3),
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


@app.post("/test-key")
async def test_key(req: TestKeyRequest):
    prov = req.provider.lower().strip()
    key = req.api_key.strip()
    model = (req.model or "").strip()

    if not key:
        raise HTTPException(status_code=400, detail="API key is empty.")

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            if prov == "serper":
                res = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": key, "Content-Type": "application/json"},
                    json={"q": "3D low poly model", "num": 1},
                )
                if res.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Serper rejected key (HTTP {res.status_code}).")
                return {"ok": True, "provider": "serper", "message": "Serper Web Search verified & active!"}

            elif prov == "gemini":
                m = normalize_gemini_model(model or "gemini-2.5-flash")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={key}"
                res = await client.post(
                    url,
                    json={
                        "contents": [{"role": "user", "parts": [{"text": "Reply with OK"}]}],
                        "generationConfig": {"maxOutputTokens": 10},
                    },
                )
                if res.status_code != 200:
                    if m != "gemini-2.5-flash":
                        url2 = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
                        res2 = await client.post(
                            url2,
                            json={
                                "contents": [{"role": "user", "parts": [{"text": "Reply with OK"}]}],
                                "generationConfig": {"maxOutputTokens": 10},
                            },
                        )
                        if res2.status_code == 200:
                            return {"ok": True, "provider": "gemini", "message": "Gemini key verified (using gemini-2.5-flash)!"}
                    raise HTTPException(status_code=400, detail=f"Gemini error ({res.status_code}): {res.text[:160]}")
                return {"ok": True, "provider": "gemini", "message": f"Gemini verified ({m})!"}

            elif prov == "openrouter":
                m = model or "google/gemini-2.5-flash"
                res = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://verdant3d.local",
                        "X-Title": "Verdant 3D Builder",
                    },
                    json={
                        "model": m,
                        "messages": [{"role": "user", "content": "Reply with OK"}],
                        "max_tokens": 10,
                    },
                )
                if res.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"OpenRouter error ({res.status_code}): {res.text[:160]}")
                return {"ok": True, "provider": "openrouter", "message": f"OpenRouter verified ({m})!"}

            elif prov == "openai":
                m = model or "gpt-4o-mini"
                res = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": m,
                        "messages": [{"role": "user", "content": "Reply with OK"}],
                        "max_tokens": 10,
                    },
                )
                if res.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"OpenAI error ({res.status_code}): {res.text[:160]}")
                return {"ok": True, "provider": "openai", "message": f"OpenAI verified ({m})!"}

            elif prov == "anthropic":
                m = model or "claude-sonnet-4-20250514"
                res = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": m,
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": "Reply with OK"}],
                    },
                )
                if res.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Anthropic error ({res.status_code}): {res.text[:160]}")
                return {"ok": True, "provider": "anthropic", "message": f"Anthropic verified ({m})!"}

            else:
                raise HTTPException(status_code=400, detail=f"Unknown provider: {prov}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/generate")
async def generate(req: GenerateRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    web_ctx = req.web_context.strip() if req.web_context else ""
    if not web_ctx and req.use_serper:
        web_ctx = await fetch_anatomical_research(prompt, override_key=req.serper_key or "")

    user_instruction = f"Subject: '{prompt}'. Build a clean, game-ready Roblox asset using 30 to 55 crisp parts."
    if web_ctx:
        user_instruction += f"\n\nReal-World Web Reference (via Serper Search):\n{web_ctx}\nUse these accurate visual proportions, structural components, and colors."

    try:
        raw_output = await call_llm_architect(
            system_prompt=MASTER_SCULPTING_SKILL,
            user_prompt=user_instruction,
            provider=req.provider or "",
            model=req.model or "",
            api_key=req.api_key or "",
        )
        result = parse_and_validate_build(raw_output, prompt)
        result["used_serper"] = bool(web_ctx)
        result["web_context"] = web_ctx
        return JSONResponse(content=result)
    except HTTPException:
        raise
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
