"""
Song Grabber: paste a social media video link. The app listens to the video's audio at several
points, identifies every song it hears, and adds them to a Spotify playlist.

Setup
1. Install ffmpeg, then: pip install -r requirements.txt
2. Spotify: create an app at https://developer.spotify.com/dashboard and add the Redirect URI
   http://127.0.0.1:5000/callback  (or https://YOUR-APP-URL/callback once hosted)
3. Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET. Optional: REDIRECT_URI, FLASK_SECRET,
   COOKIES_FILE (browser cookies, helps with Instagram), MAX_SAMPLES (default 8).
4. Recognition is free via Shazam (shazamio, no key). If AUDD_API_TOKEN is set, AudD is used instead.
5. python app.py -> http://127.0.0.1:5000
"""
import asyncio
import io
import math
import os
import secrets
import subprocess
import tempfile
import wave
from datetime import timedelta
from pathlib import Path

import requests
import spotipy
from flask import Flask, jsonify, redirect, request, session
from spotipy.cache_handler import FlaskSessionCacheHandler
from spotipy.oauth2 import SpotifyOAuth
from yt_dlp import YoutubeDL
from yt_dlp.utils import match_filter_func

REDIRECT_URI = os.environ.get("REDIRECT_URI", "http://127.0.0.1:5000/callback")
SCOPE = "playlist-read-private playlist-modify-private playlist-modify-public"
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", 8))
WINDOW_SECS, STEP_SECS, MAX_AUDIO_SECS = 10, 15, 300

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET") or secrets.token_hex(16)
app.permanent_session_lifetime = timedelta(days=60)
app.config.update(SESSION_COOKIE_SECURE=REDIRECT_URI.startswith("https"), SESSION_COOKIE_SAMESITE="Lax")


def oauth():
    return SpotifyOAuth(redirect_uri=REDIRECT_URI, scope=SCOPE, cache_handler=FlaskSessionCacheHandler(session))


def spotify():
    auth = oauth()
    if not auth.validate_token(auth.cache_handler.get_cached_token()):
        return None
    return spotipy.Spotify(auth_manager=auth)


def fetch_audio(url, folder):
    """Fetch the video's audio (kept only in a temp folder) and convert to 16 kHz mono wav."""
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(Path(folder) / "src.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "match_filter": match_filter_func("duration < 900"),
    }
    if os.environ.get("COOKIES_FILE"):
        opts["cookiefile"] = os.environ["COOKIES_FILE"]
    with YoutubeDL(opts) as ydl:
        ydl.download([url])
    sources = list(Path(folder).glob("src.*"))
    if not sources:
        raise RuntimeError("video is unavailable or longer than 15 minutes")
    wav = Path(folder) / "clip.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(sources[0]), "-t", str(MAX_AUDIO_SECS), "-ar", "16000", "-ac", "1", str(wav)],
        check=True, capture_output=True,
    )
    return wav


def slice_windows(wav_path):
    """Cut short clips spread evenly across the audio."""
    clips = []
    with wave.open(str(wav_path)) as w:
        rate, total = w.getframerate(), w.getnframes()
        win = rate * WINDOW_SECS
        n = min(MAX_SAMPLES, max(1, math.ceil(total / rate / STEP_SECS)))
        last_start = max(total - win, 0)
        for i in range(n):
            start = 0 if n == 1 else int(i * last_start / (n - 1))
            w.setpos(start)
            frames = w.readframes(min(win, total - start))
            buf = io.BytesIO()
            with wave.open(buf, "wb") as o:
                o.setnchannels(1)
                o.setsampwidth(2)
                o.setframerate(rate)
                o.writeframes(frames)
            clips.append(buf.getvalue())
    return clips


def recognize(wav_bytes):
    """Return {'title','artist', optional 'spotify': {'uri'}} or None."""
    token = os.environ.get("AUDD_API_TOKEN")
    if token:
        r = requests.post(
            "https://api.audd.io/",
            data={"api_token": token, "return": "spotify"},
            files={"file": ("clip.wav", wav_bytes)},
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("result")

    from shazamio import Shazam  # free, no key

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.wav"
        path.write_bytes(wav_bytes)
        shazam = Shazam()
        fn = getattr(shazam, "recognize", None) or shazam.recognize_song
        out = asyncio.run(fn(str(path)))
    track = (out or {}).get("track")
    return {"title": track["title"], "artist": track["subtitle"]} if track else None


def spotify_uri(sp, hit):
    uri = (hit.get("spotify") or {}).get("uri")
    if uri:
        return uri
    for q in (f"track:{hit['title']} artist:{hit['artist']}", f"{hit['title']} {hit['artist']}"):
        found = sp.search(q=q, type="track", limit=1)["tracks"]["items"]
        if found:
            return found[0]["uri"]
    return None


@app.route("/login")
def login():
    return redirect(oauth().get_authorize_url())


@app.route("/callback")
def callback():
    session.permanent = True
    oauth().get_access_token(request.args.get("code"), as_dict=False)
    return redirect("/")


@app.route("/api/playlists")
def playlists():
    sp = spotify()
    if not sp:
        return jsonify(error="Not connected"), 401
    me = sp.current_user()["id"]
    items, page = [], sp.current_user_playlists(limit=50)
    while page:
        items += [
            {"id": p["id"], "name": p["name"]}
            for p in page["items"]
            if p and (p["owner"]["id"] == me or p.get("collaborative"))
        ]
        page = sp.next(page) if page["next"] else None
    return jsonify(playlists=items)


@app.route("/api/add", methods=["POST"])
def add():
    sp = spotify()
    if not sp:
        return jsonify(error="Connect Spotify first."), 401
    body = request.get_json(force=True)
    url, playlist_id = body.get("url", "").strip(), body.get("playlist_id")
    if not url or not playlist_id:
        return jsonify(error="Paste a video link and pick a playlist."), 400

    hits, failures = {}, 0
    with tempfile.TemporaryDirectory() as tmp:
        try:
            clips = slice_windows(fetch_audio(url, tmp))
        except subprocess.CalledProcessError:
            return jsonify(error="That video has no audio I can read."), 422
        except Exception as e:
            return jsonify(error=f"Couldn't get that video: {str(e)[:200]}"), 422
        for clip in clips:
            try:
                hit = recognize(clip)
            except Exception:
                failures += 1
                continue
            if hit:
                hits.setdefault((hit["title"].lower(), hit["artist"].lower()), hit)

    if not hits:
        if failures == len(clips):
            return jsonify(error="Song recognition failed. Try again in a moment."), 502
        return jsonify(error="No songs recognized. The audio may be original sound or too quiet."), 404

    page, existing = sp.playlist_items(playlist_id, fields="items(track(uri)),next", limit=100), set()
    while page:
        existing |= {i["track"]["uri"] for i in page["items"] if i.get("track")}
        page = sp.next(page) if page.get("next") else None

    results, to_add = [], []
    for hit in hits.values():
        item = {"title": hit["title"], "artist": hit["artist"]}
        uri = spotify_uri(sp, hit)
        if not uri:
            item["status"] = "not_found"
        elif uri in existing or uri in to_add:
            item["status"] = "duplicate"
        else:
            item["status"] = "added"
            to_add.append(uri)
        results.append(item)
    if to_add:
        sp.playlist_add_items(playlist_id, to_add)
    return jsonify(results=results)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Song Grabber</title>
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#3b2fd9">
<link rel="apple-touch-icon" href="/icon.svg">
<style>
:root{--bg:#f3efe6;--ink:#1d1b2e;--mute:#6a6678;--line:#d9d2c3;--accent:#3b2fd9;--ok:#16794c;--bad:#b3261e}
@media(prefers-color-scheme:dark){:root{--bg:#15141f;--ink:#efecf7;--mute:#9a96ac;--line:#2e2c40;--accent:#8f86ff;--ok:#5fd39b;--bad:#ff8a80}}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;background:var(--bg);color:var(--ink);font:17px/1.5 Georgia,'Iowan Old Style',serif;display:grid;place-items:center;padding:24px}
main{width:min(560px,100%)}
h1{font-size:2.6rem;line-height:1.05;margin:0 0 .4rem;letter-spacing:-.02em}
p.lead{color:var(--mute);margin:0 0 2rem}
label{display:block;font-size:.9rem;color:var(--mute);margin:1.2rem 0 .35rem;font-family:system-ui,sans-serif}
input,select,button,.btn{font:inherit;font-family:system-ui,sans-serif;width:100%;padding:.8rem .9rem;border:1.5px solid var(--line);border-radius:6px;background:transparent;color:var(--ink)}
input:focus,select:focus,button:focus-visible,.btn:focus-visible{outline:3px solid var(--accent);outline-offset:2px}
button,.btn{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600;cursor:pointer;margin-top:1.4rem;display:block;text-align:center;text-decoration:none}
button:disabled{opacity:.6;cursor:wait}
#out{margin-top:1.4rem;min-height:3rem;font-family:system-ui,sans-serif}
#out ul{list-style:none;margin:0;padding:0}
#out li{padding:.7rem 0;border-top:1px solid var(--line)}
#out li strong{display:block;font:1.2rem Georgia,serif}
.ok{color:var(--ok)}.bad{color:var(--bad)}.mute{color:var(--mute)}
</style></head>
<body><main>
<h1>Song Grabber</h1>
<p class="lead">Paste a video link. We find the songs in it and add them to your playlist.</p>
<div id="connect" hidden><a class="btn" href="/login">Connect Spotify</a></div>
<div id="app" hidden>
  <label for="url">Video link</label>
  <input id="url" type="url" placeholder="https://www.tiktok.com/@user/video/…" autocomplete="off">
  <label for="pl">Playlist</label>
  <select id="pl"></select>
  <button id="go">Find songs and add them</button>
</div>
<div id="out" role="status" aria-live="polite"></div>
</main>
<script>
const $=id=>document.getElementById(id), out=$('out');
function say(text,cls){out.className=cls||'';out.replaceChildren(document.createTextNode(text))}
const LABEL={added:['added','ok'],duplicate:['already in playlist','mute'],not_found:['not found on Spotify','bad']};

async function init(){
  const r=await fetch('/api/playlists');
  if(r.status===401){$('connect').hidden=false;return}
  const {playlists}=await r.json();
  $('pl').replaceChildren(...playlists.map(p=>new Option(p.name,p.id)));
  const last=localStorage.getItem('pl'); if(last)$('pl').value=last;
  $('app').hidden=false;
}
$('go').onclick=async()=>{
  const b=$('go'); b.disabled=true; say('Fetching the video and listening… this can take up to a minute.');
  localStorage.setItem('pl',$('pl').value);
  try{
    const r=await fetch('/api/add',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({url:$('url').value,playlist_id:$('pl').value})});
    const d=await r.json();
    if(!r.ok)return say(d.error,'bad');
    const ul=document.createElement('ul');
    for(const s of d.results){
      const li=document.createElement('li'), t=document.createElement('strong'), m=document.createElement('span');
      t.textContent=s.title; m.textContent=s.artist+' · '+LABEL[s.status][0]; m.className=LABEL[s.status][1];
      li.append(t,m); ul.append(li);
    }
    out.className=''; out.replaceChildren(ul); $('url').value='';
  }catch(e){say('Something went wrong. Check that the app is still running.','bad')}
  finally{b.disabled=false}
};
init().then(()=>{
  const q=new URLSearchParams(location.search), m=((q.get('url')||'')+' '+(q.get('text')||'')).match(/https?:\/\/\S+/);
  if(!m)return;
  $('url').value=m[0]; history.replaceState(null,'','/');
  if(!$('app').hidden&&localStorage.getItem('pl'))$('go').click();
});
if('serviceWorker' in navigator)navigator.serviceWorker.register('/sw.js');
</script></body></html>"""


@app.route("/")
def home():
    return PAGE


@app.route("/manifest.json")
def manifest():
    return jsonify(
        name="Song Grabber", short_name="Song Grabber", start_url="/", display="standalone",
        background_color="#f3efe6", theme_color="#3b2fd9",
        icons=[{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any"}],
        share_target={"action": "/", "method": "GET", "params": {"title": "title", "text": "text", "url": "url"}},
    )


@app.route("/sw.js")
def sw():
    return app.response_class("self.addEventListener('fetch',()=>{});", mimetype="text/javascript")


@app.route("/icon.svg")
def icon():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><rect width="512" height="512" fill="#3b2fd9"/>'
           '<path d="M200 140v190a50 50 0 1 0 30 46V210l130-30v110a50 50 0 1 0 30 46V110z" fill="#f3efe6"/></svg>')
    return app.response_class(svg, mimetype="image/svg+xml")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
