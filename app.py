"""
Song Grabber: paste a social media video link. The app listens to the video's audio at several
points, identifies every song it hears, and adds them to a Spotify playlist.

Setup
1. Install ffmpeg, then: pip install -r requirements.txt
2. Spotify: create an app at https://developer.spotify.com/dashboard and add the Redirect URI
   http://127.0.0.1:5000/callback  (or https://YOUR-APP-URL/callback once hosted)
3. Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET. Optional: REDIRECT_URI, FLASK_SECRET,
   COOKIES_FILE (path to a Netscape cookies.txt; needed for most Instagram reels),
   MAX_SAMPLES (default 60), MAX_UPLOAD_MB (default 100).
4. Recognition is free via Shazam (shazamio, no key). If AUDD_API_TOKEN is set, AudD is used instead.
5. python app.py -> http://127.0.0.1:5000
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
import secrets
import shutil
import threading
import time
import subprocess
import tempfile
import wave
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests
import spotipy
from flask import Flask, jsonify, redirect, request, session
from spotipy.cache_handler import FlaskSessionCacheHandler
from spotipy.oauth2 import SpotifyOAuth
from yt_dlp import YoutubeDL
from yt_dlp.utils import match_filter_func

REDIRECT_URI = os.environ.get("REDIRECT_URI", "http://127.0.0.1:5000/callback")
SCOPE = "playlist-read-private playlist-modify-private playlist-modify-public"
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", 60))
WORKERS = int(os.environ.get("RECOGNIZE_WORKERS", 4))
CLIP_TIMEOUT = 25  # seconds to wait for one clip
BUDGET_SECS = int(os.environ.get("RECOGNIZE_BUDGET_SECS", 240))  # stop listening after this long
WINDOW_SECS, MAX_AUDIO_SECS = 10, 600
STEPS = {"fast": 15, "balanced": 10, "thorough": 7}  # seconds between the starts of 10 s clips

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET") or secrets.token_hex(16)
app.permanent_session_lifetime = timedelta(days=60)
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", 100))
app.config.update(
    SESSION_COOKIE_SECURE=REDIRECT_URI.startswith("https"),
    SESSION_COOKIE_SAMESITE="Lax",
    MAX_CONTENT_LENGTH=MAX_UPLOAD_MB * 1024 * 1024,
)


def oauth():
    return SpotifyOAuth(redirect_uri=REDIRECT_URI, scope=SCOPE, cache_handler=FlaskSessionCacheHandler(session))


def spotify():
    auth = oauth()
    token = auth.validate_token(auth.cache_handler.get_cached_token())
    if not token:
        return None
    return spotipy.Spotify(auth=token["access_token"])


def to_wav(src, folder):
    """Convert any audio/video file to 16 kHz mono wav (first few minutes only)."""
    wav = Path(folder) / "clip.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn", "-t", str(MAX_AUDIO_SECS), "-ar", "16000", "-ac", "1", str(wav)],
        check=True, capture_output=True,
    )
    return wav


def clean_url(url):
    u = urlparse(url)
    if "instagram.com" in u.netloc:  # drop tracking params like ?igsh=
        return urlunparse(u._replace(query="", fragment=""))
    return url


def explain(url, err):
    msg = str(err)
    low = msg.lower()
    if "instagram" in url and any(k in low for k in ("login", "cookies", "rate-limit", "empty media", "not available")):
        return ("Instagram blocked the download (it usually requires being logged in). "
                "Add Instagram cookies on the server, or screen-record the reel and use the Upload tab.")
    return msg[:300]


def fetch_audio(url, folder):
    """Fetch a video's audio into a temp folder and convert it to wav."""
    url = clean_url(url)
    opts = {
        "format": "bestaudio[abr<=?130]/worstaudio/worst",  # small files download faster
        "outtmpl": str(Path(folder) / "src.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "match_filter": match_filter_func("duration < 900"),
    }
    cookies = os.environ.get("COOKIES_FILE")
    if cookies and Path(cookies).is_file():
        # yt-dlp rewrites its cookie file, and hosts mount secret files read-only, so use a copy
        copy = Path(folder) / "cookies.txt"
        copy.write_text(Path(cookies).read_text())
        opts["cookiefile"] = str(copy)
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise RuntimeError(explain(url, e)) from e
    sources = list(Path(folder).glob("src.*"))
    if not sources:
        raise RuntimeError("video is unavailable or longer than 15 minutes")
    return to_wav(sources[0], folder)


def slice_windows(wav_path, step):
    """Cut short clips spread evenly across the audio."""
    clips = []
    with wave.open(str(wav_path)) as w:
        rate, total = w.getframerate(), w.getnframes()
        win = rate * WINDOW_SECS
        last_start = max(total - win, 0)
        n = min(MAX_SAMPLES, math.ceil(last_start / (rate * step)) + 1)
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


def _track(out):
    track = (out or {}).get("track")
    return {"title": track["title"], "artist": track["subtitle"]} if track else None


async def _shazam_all(clips, job):
    from shazamio import Shazam  # free, no key

    shazam, gate = Shazam(), asyncio.Semaphore(WORKERS)
    call = getattr(shazam, "recognize", None) or shazam.recognize_song
    deadline = time.monotonic() + BUDGET_SECS

    async def one(clip, folder, i):
        try:
            path = Path(folder) / f"clip{i}.wav"
            path.write_bytes(clip)
            err = None
            async with gate:
                if time.monotonic() > deadline:
                    job["skipped"] = job.get("skipped", 0) + 1
                    return TimeoutError("time budget used up")
                for _ in range(2):  # one retry for flaky responses
                    try:
                        return _track(await asyncio.wait_for(call(str(path)), CLIP_TIMEOUT))
                    except Exception as e:
                        err = e
                        await asyncio.sleep(1)
            return err
        finally:
            job["done"] += 1

    with tempfile.TemporaryDirectory() as folder:
        return await asyncio.gather(*(one(c, folder, i) for i, c in enumerate(clips)))


def recognize_all(clips, job):
    """Recognize clips in parallel. One entry per clip, in order:
    a hit dict, None (no match) or an Exception (that clip failed)."""
    token = os.environ.get("AUDD_API_TOKEN")
    if token:
        def one(clip):
            try:
                r = requests.post(
                    "https://api.audd.io/",
                    data={"api_token": token, "return": "spotify"},
                    files={"file": ("clip.wav", clip)},
                    timeout=CLIP_TIMEOUT,
                )
                r.raise_for_status()
                return r.json().get("result")
            except Exception as e:
                return e
            finally:
                job["done"] += 1

        with ThreadPoolExecutor(WORKERS) as pool:
            return list(pool.map(one, clips))
    return asyncio.run(_shazam_all(clips, job))


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


class JobError(Exception):
    """A problem worth showing to the person as-is."""


JOBS = {}  # kept in memory; fine because the server runs a single worker process


def start_job(target, *args):
    now = time.time()
    for old in [k for k, j in JOBS.items() if now - j["t"] > 3600]:
        del JOBS[old]
    job_id = secrets.token_urlsafe(8)
    job = JOBS[job_id] = {"state": "running", "stage": "Starting", "done": 0, "total": 0, "t": now, "timings": {}}

    def run():
        try:
            job["results"] = target(job, *args)
            job["timings"]["total"] = round(time.time() - job["t"], 1)
            print("job finished:", job["timings"], flush=True)  # visible in Render Logs
            job["state"] = "done"
        except JobError as e:
            job.update(state="error", error=str(e))
        except Exception as e:
            job.update(state="error", error=f"Unexpected error: {str(e)[:200]}")

    threading.Thread(target=run, daemon=True).start()
    return job_id


def safe_uri(sp, hit):
    try:
        return spotify_uri(sp, hit)
    except Exception:
        return None


def identify_and_add(job, sp, playlist_id, wav, step):
    clips = slice_windows(wav, step)
    job.update(stage="Listening", total=len(clips), done=0)
    t = time.monotonic()
    hits, failures = {}, 0
    for hit in recognize_all(clips, job):
        if isinstance(hit, Exception):
            failures += 1
        elif hit:
            hits.setdefault((hit["title"].lower(), hit["artist"].lower()), hit)
    job["timings"]["listen"] = round(time.monotonic() - t, 1)

    if not hits:
        if failures == len(clips):
            raise JobError("Song recognition failed or timed out. Try again in a moment.")
        raise JobError("No songs recognized. The audio may be original sound or too quiet.")

    job["stage"] = "Adding to Spotify"
    t = time.monotonic()
    page, existing = sp.playlist_items(playlist_id, fields="items(track(uri)),next", limit=100), set()
    while page:
        existing |= {i["track"]["uri"] for i in page["items"] if i.get("track")}
        page = sp.next(page) if page.get("next") else None

    found = list(hits.values())
    with ThreadPoolExecutor(4) as pool:  # look the songs up on Spotify in parallel
        uris = list(pool.map(lambda h: safe_uri(sp, h), found))

    results, to_add = [], []
    for hit, uri in zip(found, uris):
        item = {"title": hit["title"], "artist": hit["artist"]}
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
    job["timings"]["spotify"] = round(time.monotonic() - t, 1)
    return results


def link_task(job, sp, playlist_id, url, step):
    with tempfile.TemporaryDirectory() as tmp:
        job["stage"] = "Fetching the video"
        t = time.monotonic()
        try:
            wav = fetch_audio(url, tmp)
        except subprocess.CalledProcessError:
            raise JobError("That video has no audio I can read.")
        except Exception as e:
            raise JobError(f"Couldn't get that video: {str(e)[:300]}")
        job["timings"]["fetch"] = round(time.monotonic() - t, 1)
        return identify_and_add(job, sp, playlist_id, wav, step)


def file_task(job, sp, playlist_id, folder, step):
    try:  # the uploaded file is deleted when this finishes
        job["stage"] = "Reading the file"
        t = time.monotonic()
        try:
            wav = to_wav(Path(folder) / "upload.bin", folder)
        except subprocess.CalledProcessError:
            raise JobError("I couldn't read audio from that file.")
        job["timings"]["fetch"] = round(time.monotonic() - t, 1)
        return identify_and_add(job, sp, playlist_id, wav, step)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


@app.route("/api/add", methods=["POST"])
def add():
    sp = spotify()
    if not sp:
        return jsonify(error="Connect Spotify first."), 401
    body = request.get_json(force=True)
    url, playlist_id = body.get("url", "").strip(), body.get("playlist_id")
    if not url or not playlist_id:
        return jsonify(error="Paste a video link and pick a playlist."), 400
    step = STEPS.get(body.get("speed"), STEPS["balanced"])
    return jsonify(job=start_job(link_task, sp, playlist_id, url, step))


@app.route("/api/upload", methods=["POST"])
def upload():
    sp = spotify()
    if not sp:
        return jsonify(error="Connect Spotify first."), 401
    video, playlist_id = request.files.get("video"), request.form.get("playlist_id")
    if not video or not playlist_id:
        return jsonify(error="Choose a file and a playlist."), 400
    folder = tempfile.mkdtemp()
    video.save(Path(folder) / "upload.bin")
    step = STEPS.get(request.form.get("speed"), STEPS["balanced"])
    return jsonify(job=start_job(file_task, sp, playlist_id, folder, step))


@app.route("/api/job/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="That job is gone (the server may have restarted). Please try again."), 404
    return jsonify({k: v for k, v in job.items() if k != "t"})


@app.errorhandler(413)
def too_big(_):
    return jsonify(error=f"That file is too large (limit {MAX_UPLOAD_MB} MB)."), 413


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
#out ul,#hl{list-style:none;margin:0;padding:0}
#out li,#hl li{padding:.7rem 0;border-top:1px solid var(--line)}
#out li strong,#hl li strong{display:block;font:1.2rem Georgia,serif}
.ok{color:var(--ok)}.bad{color:var(--bad)}.mute{color:var(--mute)}
.tabs{display:flex;gap:.5rem}.tab{margin:0;background:transparent;color:var(--ink);border-color:var(--line)}
.tab.on{background:var(--accent);border-color:var(--accent);color:#fff}
h2{font-size:1.4rem;margin:2.4rem 0 .4rem}#clear{margin-top:1rem;font-weight:400;color:var(--mute)}
</style></head>
<body><main>
<h1>Song Grabber</h1>
<p class="lead">Paste a video link. We find the songs in it and add them to your playlist.</p>
<div id="connect" hidden><a class="btn" href="/login">Connect Spotify</a></div>
<div id="app" hidden>
  <div class="tabs" role="tablist">
    <button type="button" id="tab-link" class="tab on" role="tab" aria-selected="true">Link</button>
    <button type="button" id="tab-file" class="tab" role="tab" aria-selected="false">Upload</button>
  </div>
  <div id="pane-link">
    <label for="url">Video link</label>
    <input id="url" type="url" placeholder="https://www.instagram.com/reel/…" autocomplete="off">
  </div>
  <div id="pane-file" hidden>
    <label for="file">Video file (not stored; deleted right after listening)</label>
    <input id="file" type="file" accept="video/*,audio/*">
  </div>
  <label for="pl">Playlist</label>
  <select id="pl"></select>
  <label for="speed">Detail</label>
  <select id="speed">
    <option value="fast">Fast (checks every 15 seconds)</option>
    <option value="balanced" selected>Balanced (every 10 seconds)</option>
    <option value="thorough">Thorough (every 7 seconds)</option>
  </select>
  <button id="go">Find songs and add them</button>
</div>
<div id="out" role="status" aria-live="polite"></div>
<section id="hist" hidden><h2>History</h2><ul id="hl"></ul><button type="button" id="clear" class="tab">Clear history</button></section>
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
let mode='link';
function setMode(m){
  mode=m; $('pane-link').hidden=m!=='link'; $('pane-file').hidden=m!=='file';
  for(const [id,on] of [['tab-link',m==='link'],['tab-file',m==='file']]){
    $(id).classList.toggle('on',on); $(id).setAttribute('aria-selected',on)}
}
$('tab-link').onclick=()=>setMode('link'); $('tab-file').onclick=()=>setMode('file');

const loadHist=()=>{try{return JSON.parse(localStorage.getItem('history')||'[]')}catch(e){return[]}};
function saveHist(list){try{localStorage.setItem('history',JSON.stringify(list.slice(0,500)))}catch(e){}}
function renderHist(){
  const h=loadHist(); $('hist').hidden=!h.length;
  $('hl').replaceChildren(...h.map(x=>{
    const li=document.createElement('li'), t=document.createElement('strong'), m=document.createElement('span');
    t.textContent=x.title;
    m.textContent=[x.artist,LABEL[x.status][0],x.playlist,x.source,new Date(x.at).toLocaleString()].join(' · ');
    m.className='mute'; li.append(t,m); return li;
  }));
}
$('clear').onclick=()=>{if(confirm('Clear the history on this device?')){saveHist([]);renderHist()}};

async function waitFor(id){
  let misses=0;
  for(;;){
    await new Promise(r=>setTimeout(r,1500));
    let r,d;
    try{r=await fetch('/api/job/'+id); d=await r.json()}
    catch(e){if(++misses>8)throw new Error('Lost connection to the server.'); continue}
    misses=0;
    if(!r.ok)throw new Error(d.error);
    if(d.state!=='running')return d;
    say(d.total?`${d.stage}… ${d.done} of ${d.total} clips checked`:d.stage+'…');
  }
}

$('go').onclick=async()=>{
  const b=$('go'), pl=$('pl'); b.disabled=true;
  try{localStorage.setItem('pl',pl.value)}catch(e){}
  try{
    let r;
    if(mode==='file'){
      const f=$('file').files[0]; if(!f)return say('Choose a video file first.','bad');
      say('Uploading…');
      const fd=new FormData(); fd.append('video',f); fd.append('playlist_id',pl.value); fd.append('speed',$('speed').value);
      r=await fetch('/api/upload',{method:'POST',body:fd});
    }else{
      if(!$('url').value.trim())return say('Paste a video link first.','bad');
      say('Starting…');
      r=await fetch('/api/add',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({url:$('url').value,playlist_id:pl.value,speed:$('speed').value})});
    }
    const start=await r.json();
    if(!r.ok)return say(start.error,'bad');
    const d=await waitFor(start.job);
    if(d.state==='error')return say(d.error,'bad');
    const box=document.createElement('div'), ul=document.createElement('ul');
    for(const s of d.results){
      const li=document.createElement('li'), t=document.createElement('strong'), m=document.createElement('span');
      t.textContent=s.title; m.textContent=s.artist+' · '+LABEL[s.status][0]; m.className=LABEL[s.status][1];
      li.append(t,m); ul.append(li);
    }
    box.append(ul);
    if(d.skipped){const n=document.createElement('p'); n.className='mute';
      n.textContent=`Stopped early: ${d.skipped} of ${d.total} clips weren't checked, so some songs may be missing.`; box.append(n)}
    if(d.timings&&d.timings.total!==undefined){const t=document.createElement('p'); t.className='mute';
      t.textContent=`Finished in ${d.timings.total}s (fetch ${d.timings.fetch}s, listening ${d.timings.listen}s, Spotify ${d.timings.spotify}s).`; box.append(t)}
    out.className=''; out.replaceChildren(box);
    const src=mode==='file'?'upload':'link', plName=pl.selectedOptions[0].text, at=Date.now();
    saveHist([...d.results.map(s=>({...s,source:src,playlist:plName,at})),...loadHist()]);
    renderHist(); $('url').value=''; $('file').value='';
  }catch(e){say(e.message||'Something went wrong.','bad')}
  finally{b.disabled=false}
};
renderHist();
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
