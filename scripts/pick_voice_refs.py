#!/usr/bin/env python3
"""Local listening page for picking voice refs and vetting generated clips.

Serves a single page on http://localhost:8077 with every sliced voice-ref
candidate (play button + editable transcript + one-click pick per character
pack) and every generated non-verbal / filler clip (play + reject). Picking
copies the candidate to the pack's voice_ref.wav and writes the voice block
into character.yaml with your corrected transcript. Rejecting a generated
clip moves it into a rejected/ subfolder and drops it from the manifest;
re-run the generator with --force to refill gaps.

Everything stays on this machine; nothing here touches the network or git.

Usage:
    python scripts/pick_voice_refs.py [--port 8077]
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKS_DIR = ROOT / "characters" / "packs"
CLIP_SETS = {
    "nonverbals": ROOT / "static" / "nonverbals",
    "fillers": ROOT / "static" / "fillers",
}
LONG_CLIP_SECONDS = 4.0


def wav_seconds(path: Path) -> float:
    import wave

    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def list_packs() -> list[dict]:
    packs = []
    for pack_dir in sorted(PACKS_DIR.iterdir()):
        cand_dir = pack_dir / "candidates"
        sheet = cand_dir / "transcripts.json"
        if not sheet.exists():
            continue
        rows = json.loads(sheet.read_text(encoding="utf-8"))
        yaml_text = (pack_dir / "character.yaml").read_text(encoding="utf-8")
        picked = (pack_dir / "voice_ref.wav").exists()
        current = re.search(r'^\s*ref_text:\s*"(.*)"', yaml_text, re.M)
        packs.append({
            "id": pack_dir.name,
            "picked": picked,
            "ref_text": current.group(1) if current else "",
            "candidates": [
                {
                    "file": r["file"],
                    "duration": r.get("duration", 0),
                    "transcript": r.get("transcript", ""),
                }
                for r in rows
                if (cand_dir / r["file"]).exists()
            ],
        })
    return packs


def list_clips() -> dict[str, list[dict]]:
    sets: dict[str, list[dict]] = {}
    for name, folder in CLIP_SETS.items():
        clips = []
        for wav in sorted(folder.glob("*.wav")):
            seconds = wav_seconds(wav)
            clips.append({
                "file": wav.name,
                "duration": round(seconds, 2),
                "long": seconds > LONG_CLIP_SECONDS,
            })
        sets[name] = clips
    return sets


def apply_pick(pack_id: str, cand_file: str, ref_text: str) -> None:
    pack_dir = PACKS_DIR / Path(pack_id).name
    src = pack_dir / "candidates" / Path(cand_file).name
    if not src.exists():
        raise FileNotFoundError(cand_file)
    shutil.copyfile(src, pack_dir / "voice_ref.wav")

    yaml_path = pack_dir / "character.yaml"
    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    # Drop the commented voice template and any previously written block.
    kept, in_block = [], False
    for line in lines:
        if re.match(r"^# Voice: needs|^# voice:|^#   ref_|^#   language:", line):
            continue
        if re.match(r"^voice:\s*$", line):
            in_block = True
            continue
        if in_block and re.match(r"^\s+\S", line):
            continue
        in_block = False
        kept.append(line)
    while kept and not kept[-1].strip():
        kept.pop()
    kept += [
        "voice:",
        "  ref_audio: voice_ref.wav",
        f"  ref_text: {json.dumps(ref_text, ensure_ascii=False)}",
        "  language: Japanese",
        "",
    ]
    yaml_path.write_text("\n".join(kept), encoding="utf-8")


def apply_reject(set_name: str, clip_file: str) -> None:
    folder = CLIP_SETS[set_name]
    wav = folder / Path(clip_file).name
    if not wav.exists():
        raise FileNotFoundError(clip_file)
    rejected = folder / "rejected"
    rejected.mkdir(exist_ok=True)
    shutil.move(str(wav), rejected / wav.name)

    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in list(manifest.items()):
        if isinstance(value, list):
            manifest[key] = [
                v for v in value
                if (v if isinstance(v, str) else v.get("file")) != wav.name
            ]
        elif isinstance(value, dict):
            for kind in list(value):
                if isinstance(value[kind], list):
                    value[kind] = [v for v in value[kind] if v != wav.name]
                    if not value[kind]:
                        del value[kind]
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Amadeus voice picks</title>
<style>
body{background:#14151a;color:#e8e6e3;font:15px/1.5 system-ui,sans-serif;
     max-width:880px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:2.2rem;
   border-bottom:1px solid #333;padding-bottom:.3rem}
.cand{display:flex;gap:.8rem;align-items:center;margin:.55rem 0;
      background:#1d1f26;border-radius:8px;padding:.55rem .8rem}
.cand.picked{outline:2px solid #4caf7d}
audio{height:32px;flex:0 0 260px}
textarea{flex:1;background:#101116;color:#e8e6e3;border:1px solid #333;
         border-radius:6px;padding:.35rem .5rem;font:inherit;resize:vertical;
         min-height:2.2rem}
button{background:#7a3040;color:#fff;border:0;border-radius:6px;
       padding:.45rem .8rem;cursor:pointer;white-space:nowrap}
button:hover{background:#95404f}
.meta{color:#8b8fa3;font-size:.82rem;white-space:nowrap}
.long{color:#e6b84c}
.done{color:#4caf7d;font-weight:600}
.note{color:#8b8fa3;font-size:.9rem}
</style></head><body>
<h1>Voice picks &amp; clip vetting</h1>
<p class="note">Pick ONE candidate per character (fix the transcript first;
whisper guesses are rough). Rejected clips move to a rejected/ folder, they
are not deleted.</p>
__BODY__
<script>
async function post(url, payload, el){
  const r = await fetch(url,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  if(r.ok){ el.closest('.cand').classList.add('picked');
            el.textContent='done'; el.disabled=true; }
  else alert(await r.text());
}
function pick(btn, pack, file){
  const text = btn.closest('.cand').querySelector('textarea').value.trim();
  if(!text){alert('Transcript is empty.');return;}
  post('/pick',{pack, file, text}, btn);
}
function reject(btn, set, file){ post('/reject',{set, file}, btn); }
</script></body></html>"""


def render_page() -> str:
    parts = []
    for pack in list_packs():
        status = ' <span class="done">voice_ref.wav set</span>' if pack["picked"] else ""
        parts.append(f"<h2>{html.escape(pack['id'])}{status}</h2>")
        for cand in pack["candidates"]:
            parts.append(
                '<div class="cand">'
                f'<audio controls preload="none" '
                f'src="/audio/packs/{pack["id"]}/{cand["file"]}"></audio>'
                f'<span class="meta">{cand["duration"]:.1f}s</span>'
                f"<textarea>{html.escape(cand['transcript'])}</textarea>"
                f'<button onclick="pick(this,\'{pack["id"]}\',\'{cand["file"]}\')">'
                "Pick</button></div>"
            )
    for set_name, clips in list_clips().items():
        parts.append(f"<h2>{set_name} (generated, vet by ear)</h2>")
        for clip in clips:
            flag = ' <span class="long">long!</span>' if clip["long"] else ""
            parts.append(
                '<div class="cand">'
                f'<audio controls preload="none" '
                f'src="/audio/{set_name}/{clip["file"]}"></audio>'
                f'<span class="meta">{clip["duration"]:.1f}s{flag}</span>'
                f'<span class="meta" style="flex:1">{clip["file"]}</span>'
                f'<button onclick="reject(this,\'{set_name}\',\'{clip["file"]}\')">'
                "Reject</button></div>"
            )
    return PAGE.replace("__BODY__", "\n".join(parts))


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, render_page().encode("utf-8"), "text/html; charset=utf-8")
            return
        m = re.match(r"^/audio/packs/([\w-]+)/([\w.-]+\.wav)$", self.path)
        if m:
            wav = PACKS_DIR / m.group(1) / "candidates" / m.group(2)
        else:
            m = re.match(r"^/audio/(nonverbals|fillers)/([\w.-]+\.wav)$", self.path)
            wav = CLIP_SETS[m.group(1)] / m.group(2) if m else None
        if wav and wav.exists():
            self._send(200, wav.read_bytes(), "audio/wav")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))
        try:
            if self.path == "/pick":
                apply_pick(payload["pack"], payload["file"], payload["text"])
            elif self.path == "/reject":
                set_name = payload["set"]
                if set_name not in CLIP_SETS:
                    raise ValueError(set_name)
                apply_reject(set_name, payload["file"])
            else:
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, b"ok", "text/plain")
        except Exception as exc:  # surfaced in the page alert
            self._send(400, str(exc).encode("utf-8"), "text/plain")

    def log_message(self, *args) -> None:  # keep the terminal quiet
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8077)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Listening page: http://localhost:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
