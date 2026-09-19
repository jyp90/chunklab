"""Record a chunklab UI demo as an animated GIF (frame-by-frame screenshots, no ffmpeg)."""
# Run from the repo root: uv run --with playwright --with pillow --with httpx python docs/images/demo_gif.py
import io, shutil, subprocess, tempfile, time
from pathlib import Path
import httpx
from PIL import Image
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[2]
WS = Path(tempfile.mkdtemp(prefix="chunklab-demo-")); shutil.rmtree(WS, ignore_errors=True); WS.mkdir(parents=True)
OUT = REPO / "docs" / "images" / "demo.gif"
PORT = 7882; BASE = f"http://127.0.0.1:{PORT}"
SAMPLE = REPO / "examples" / "quickstart" / "docs" / "refund-policy.md"
QJSON = REPO / "examples" / "quickstart" / "questions.json"
W = 960
frames: list[tuple[Image.Image, int]] = []

def snap(page, ms, full=False):
    png = page.screenshot(full_page=full)
    im = Image.open(io.BytesIO(png)).convert("RGB")
    if full and im.height > 1400:
        im = im.crop((0, 0, im.width, 1400))
    im = im.resize((W, round(im.height * W / im.width)), Image.LANCZOS)
    frames.append((im, ms))

srv = subprocess.Popen(["uv","run","chunklab","ui","--workspace",str(WS),"--no-browser","--port",str(PORT)], cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(100):
        try:
            if httpx.get(f"{BASE}/health", timeout=1).status_code == 200: break
        except Exception: time.sleep(0.2)
    with sync_playwright() as p:
        b = p.chromium.launch(); page = b.new_page(viewport={"width": 1280, "height": 800})
        page.goto(BASE + "/"); snap(page, 1200)
        page.set_input_files('input[type=file]', str(SAMPLE))
        page.click('button:has-text("Upload")'); page.wait_for_selector('#doc-list a:has-text("refund-policy")')
        page.click('#doc-list a:has-text("refund-policy")'); page.wait_for_selector("pre#doc"); snap(page, 1500)
        # drag-select a passage -> toolbar -> add question
        page.evaluate("""() => {const pre=document.getElementById('doc');const t=pre.textContent;const s=t.indexOf('International shipping');const e=t.indexOf('7 to 14 days.')+13;
            const walker=document.createTreeWalker(pre,NodeFilter.SHOW_TEXT);let acc=0,n,sn,so,en,eo;while((n=walker.nextNode())){const L=n.nodeValue.length;if(sn===undefined&&s<acc+L){sn=n;so=s-acc;}if(e<=acc+L){en=n;eo=e-acc;break;}acc+=L;}
            const r=document.createRange();r.setStart(sn,so);r.setEnd(en,eo);const sel=window.getSelection();sel.removeAllRanges();sel.addRange(r);document.dispatchEvent(new Event('selectionchange'));}""")
        page.wait_for_selector("#sel-toolbar:not([hidden])"); snap(page, 1200)
        page.type("#new-q-text", "How long does international shipping take?", delay=25); snap(page, 1500)
        page.click('#sel-toolbar button:has-text("Add question")'); page.wait_for_selector('#question-list li[data-qid]'); snap(page, 1200)
        # bring in the rest of the quickstart questions
        page.set_input_files('input[name="file"]', str(QJSON))
        page.click('button:has-text("Import")'); page.wait_for_function("document.querySelectorAll('#question-list li[data-qid]').length >= 5")
        page.click('#question-list a.q-link >> nth=1'); page.wait_for_selector("pre#doc mark.gold"); snap(page, 1800)

        page.goto(BASE + "/experiment"); snap(page, 1000)
        page.select_option('select[name="embedders"]', ["fake"])
        page.fill('input[name="recursive_chunk_size"]', "128, 256, 512")
        page.fill('input[name="recursive_overlap"]', "0, 32")
        page.fill('input[name="top_k"]', "3"); snap(page, 1800)
        page.click('#exp-form button.primary:has-text("Run")')
        page.wait_for_url(f"{BASE}/results/*", timeout=60000); page.wait_for_selector("tr.recommended"); snap(page, 2500)
        qid = page.eval_on_selector('select[name="qid"] option:nth-child(2)', "o => o.value")
        page.select_option('select[name="qid"]', qid); page.wait_for_selector("#question-detail pre#doc")
        page.locator("#question-detail").scroll_into_view_if_needed(); snap(page, 2500)
        page.click('tr.recommended button:has-text("code")'); page.wait_for_selector("#snippet pre.code")
        page.click('#snippet .tabs button:has-text("langchain")')
        page.wait_for_function("document.querySelector('#snippet pre.code').innerText.includes('langchain')")
        page.locator("#snippet").scroll_into_view_if_needed(); snap(page, 3000)
        b.close()
finally:
    srv.terminate()

H = max(im.height for im, _ in frames)
canvas = []
for im, ms in frames:
    c = Image.new("RGB", (W, H), (255, 255, 255)); c.paste(im, (0, 0)); canvas.append((c.quantize(colors=128, method=Image.MEDIANCUT), ms))
canvas[0][0].save(OUT, save_all=True, append_images=[c for c, _ in canvas[1:]], duration=[ms for _, ms in canvas], loop=0, optimize=True)
print(OUT, OUT.stat().st_size // 1024, "KB", len(frames), "frames", f"{W}x{H}")
