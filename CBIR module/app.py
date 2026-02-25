"""
========================================================
  Oral Cancer CBIR — Gradio Web Application
========================================================

Loads the pre-built cbir_index.npz and launches a local
web interface for image-similarity search.

USAGE:
  python run_app.py
  python run_app.py --top_k_benign 8 --top_k_malignant 8
  python run_app.py --device cuda --port 7860

PREREQUISITES:
  Run build_index.py first to generate cbir_index.npz.
"""

import os
import sys
import ssl
import argparse
import numpy as np
from pathlib import Path
import urllib.request as _urllib_request

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image
    from sklearn.metrics.pairwise import cosine_similarity
    import gradio as gr
except ImportError as e:
    print(f"\n[!] Missing dependency: {e}")
    print("    Run:  pip install torch torchvision pillow scikit-learn gradio\n")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
INDEX_PATH = r"CBIR module\cbir_index.npz"
IMG_SIZE   = 224


# ══════════════════════════════════════════════════════════════════════════════
#  SSL helpers
# ══════════════════════════════════════════════════════════════════════════════
def _disable_ssl_verification():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    opener = _urllib_request.build_opener(_urllib_request.HTTPSHandler(context=ctx))
    _urllib_request.install_opener(opener)

def _restore_ssl():
    _urllib_request.install_opener(_urllib_request.build_opener())


# ══════════════════════════════════════════════════════════════════════════════
#  Feature Extractor
# ══════════════════════════════════════════════════════════════════════════════
class MobileNetV2Extractor:
    """MobileNetV2 → 1280-d L2-normalised embedding."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        print("  Loading MobileNetV2 weights (may download ~14 MB on first run)...")
        try:
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        except Exception as e:
            if "CERTIFICATE" in str(e).upper() or "SSL" in str(e).upper():
                print("  [!] SSL error — retrying with verification disabled...")
                _disable_ssl_verification()
                try:
                    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                finally:
                    _restore_ssl()
            else:
                raise

        self.model = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        ).to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        emb = self.model(tensor).squeeze().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
#  Load Index
# ══════════════════════════════════════════════════════════════════════════════
def load_index() -> dict:
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"Index '{INDEX_PATH}' not found.\n"
            "Run build_index.py first to generate it."
        )
    data   = np.load(INDEX_PATH, allow_pickle=True)
    total  = len(data["paths"])
    labels = data["labels"].tolist()
    print(f"  ✓ Loaded index: {total} images  "
          f"({labels.count('benign')} benign, {labels.count('malignant')} malignant)")
    return dict(
        embeddings = data["embeddings"],
        paths      = data["paths"],
        labels     = data["labels"],
        sources    = data["sources"] if "sources" in data.files
                     else np.array(["unknown"] * total),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Query
# ══════════════════════════════════════════════════════════════════════════════
def query_image_split(
    query_img: Image.Image,
    index: dict,
    extractor: MobileNetV2Extractor,
    top_k_benign: int = 5,
    top_k_malignant: int = 5,
) -> tuple[list[dict], list[dict]]:
    query_emb = extractor.extract(query_img)
    sims      = cosine_similarity(query_emb.reshape(1, -1), index["embeddings"])[0]
    labels    = index["labels"]
    sources   = index.get("sources", np.array(["unknown"] * len(index["paths"])))

    def top_for_label(label: str, top_k: int) -> list[dict]:
        mask    = np.where(labels == label)[0]
        sub_sim = sims[mask]
        ranked  = mask[np.argsort(sub_sim)[::-1][:top_k]]
        return [
            {
                "rank":       rank,
                "path":       str(index["paths"][idx]),
                "label":      str(labels[idx]),
                "source":     str(sources[idx]),
                "similarity": float(sims[idx]),
                "distance":   float(1 - sims[idx]),
            }
            for rank, idx in enumerate(ranked, start=1)
        ]

    return top_for_label("benign", top_k_benign), top_for_label("malignant", top_k_malignant)


# ══════════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ══════════════════════════════════════════════════════════════════════════════
def launch_app(
    extractor: MobileNetV2Extractor,
    index: dict,
    top_k_benign: int = 5,
    top_k_malignant: int = 5,
    host: str = "127.0.0.1",
    port: int = 7860,
):
    import base64
    import io

    def pil_to_b64(img: Image.Image, max_side: int = 800) -> str:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()

    def results_to_html(
        benign_res: list[dict],
        malignant_res: list[dict],
        query_img: Image.Image,
    ) -> str:
        query_b64 = pil_to_b64(query_img)

        def card_html(r: dict, card_id: str) -> str:
            try:
                img   = Image.open(r["path"]).convert("RGB")
                img_b = pil_to_b64(img)
            except Exception:
                img_b = ""
            label_cls = r["label"]
            return f"""
            <div class="card {label_cls}" id="{card_id}"
                 onclick="openModal('{img_b}','{query_b64}','{r['label'].upper()}',
                                    '{r['similarity']:.4f}','{r['distance']:.4f}',
                                    '{r['source']}','{Path(r['path']).name}')">
              <div class="card-img-wrap">
                <img src="data:image/jpeg;base64,{img_b}" alt="match"/>
                <div class="card-badge">{r['label'].upper()}</div>
              </div>
              <div class="card-meta">
                <span class="rank">#{r['rank']}</span>
                <span class="sim">sim {r['similarity']:.4f}</span>
              </div>
            </div>"""

        benign_cards    = "".join(card_html(r, f"b{r['rank']}") for r in benign_res)
        malignant_cards = "".join(card_html(r, f"m{r['rank']}") for r in malignant_res)

        html = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&display=swap');

  :root {{
    --bg:       #0d0f14;
    --surface:  #161921;
    --border:   #252a35;
    --benign:   #00e5a0;
    --malignant:#ff4060;
    --text:     #e8ecf5;
    --muted:    #6b7592;
    --radius:   12px;
  }}

  #cbir-root * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  #cbir-root {{
    font-family: 'DM Mono', monospace;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    border-radius: 16px;
    min-height: 400px;
  }}

  #cbir-root h2 {{
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}

  #cbir-root h2 span {{
    font-size: 18px;
    font-weight: 800;
    margin-right: 10px;
    letter-spacing: 0;
  }}

  .section {{ margin-bottom: 32px; }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 12px;
  }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    cursor: pointer;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  }}
  .card:hover {{
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 12px 32px rgba(0,0,0,0.5);
  }}
  .card.benign:hover    {{ border-color: var(--benign);    box-shadow: 0 8px 28px rgba(0,229,160,0.25); }}
  .card.malignant:hover {{ border-color: var(--malignant); box-shadow: 0 8px 28px rgba(255,64,96,0.25); }}

  .card-img-wrap {{
    position: relative;
    width: 100%;
    aspect-ratio: 1;
    overflow: hidden;
    background: #0a0c10;
  }}
  .card-img-wrap img {{
    width: 100%; height: 100%;
    object-fit: cover;
    display: block;
    transition: transform .25s ease;
  }}
  .card:hover .card-img-wrap img {{ transform: scale(1.07); }}

  .card-badge {{
    position: absolute;
    top: 8px; left: 8px;
    font-family: 'Syne', sans-serif;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: .12em;
    padding: 3px 8px;
    border-radius: 20px;
    pointer-events: none;
  }}
  .benign    .card-badge {{ background: rgba(0,229,160,.18); color: var(--benign);    border: 1px solid rgba(0,229,160,.4); }}
  .malignant .card-badge {{ background: rgba(255,64,96,.18); color: var(--malignant); border: 1px solid rgba(255,64,96,.4); }}

  .card-meta {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px;
    font-size: 10px;
    color: var(--muted);
    border-top: 1px solid var(--border);
  }}
  .card-meta .rank {{ font-weight: 500; color: var(--text); }}
  .card-meta .sim  {{ font-size: 9px; }}

  /* Modal */
  #cbir-modal {{
    display: none;
    position: fixed;
    inset: 0;
    z-index: 99999;
    background: rgba(0,0,0,0.88);
    backdrop-filter: blur(8px);
    align-items: center;
    justify-content: center;
  }}
  #cbir-modal.open {{ display: flex; }}

  .modal-box {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 28px;
    max-width: 92vw;
    max-height: 90vh;
    overflow-y: auto;
    width: 860px;
    position: relative;
    animation: modalIn .22s ease;
  }}
  @keyframes modalIn {{
    from {{ opacity: 0; transform: scale(.94) translateY(12px); }}
    to   {{ opacity: 1; transform: scale(1) translateY(0); }}
  }}

  .modal-close {{
    position: absolute;
    top: 16px; right: 20px;
    background: var(--border);
    border: none;
    color: var(--muted);
    font-size: 18px;
    width: 36px; height: 36px;
    border-radius: 50%;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: background .15s, color .15s;
    line-height: 1;
  }}
  .modal-close:hover {{ background: #ff4060; color: #fff; }}

  .modal-title {{
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
  }}

  .modal-images {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
  }}

  .modal-img-panel {{
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}

  .modal-img-panel label {{
    font-size: 10px;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--muted);
  }}

  .modal-img-panel img {{
    width: 100%;
    border-radius: 12px;
    border: 1px solid var(--border);
    object-fit: contain;
    max-height: 340px;
    background: #0a0c10;
  }}

  .modal-stats {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 10px;
    border-top: 1px solid var(--border);
    padding-top: 16px;
  }}

  .stat-block {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
  }}
  .stat-block .skey {{
    font-size: 9px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
  }}
  .stat-block .sval {{
    font-size: 15px;
    font-weight: 500;
    color: var(--text);
  }}
  .stat-block.hl-benign    .sval {{ color: var(--benign); }}
  .stat-block.hl-malignant .sval {{ color: var(--malignant); }}

  .empty {{
    color: var(--muted);
    font-size: 13px;
    padding: 32px 0;
    text-align: center;
  }}
</style>

<!-- Modal -->
<div id="cbir-modal" onclick="if(event.target===this)closeModal()">
  <div class="modal-box">
    <button class="modal-close" onclick="closeModal()">✕</button>
    <div class="modal-title">Image Comparison</div>
    <div class="modal-images">
      <div class="modal-img-panel">
        <label>Query Image</label>
        <img id="modal-query" src="" alt="query"/>
      </div>
      <div class="modal-img-panel">
        <label>Matched Image — <span id="modal-label-span" style="color:inherit"></span></label>
        <img id="modal-match" src="" alt="match"/>
      </div>
    </div>
    <div class="modal-stats">
      <div class="stat-block" id="modal-label-block">
        <div class="skey">Label</div>
        <div class="sval" id="modal-label-val">—</div>
      </div>
      <div class="stat-block">
        <div class="skey">Similarity</div>
        <div class="sval" id="modal-sim">—</div>
      </div>
      <div class="stat-block">
        <div class="skey">Distance</div>
        <div class="sval" id="modal-dist">—</div>
      </div>
      <div class="stat-block">
        <div class="skey">Source Folder</div>
        <div class="sval" id="modal-source" style="font-size:11px">—</div>
      </div>
      <div class="stat-block">
        <div class="skey">Filename</div>
        <div class="sval" id="modal-fname" style="font-size:10px;word-break:break-all">—</div>
      </div>
    </div>
  </div>
</div>

<div id="cbir-root">
  <div class="section">
    <h2><span>🟢</span>Top {len(benign_res)} Benign Matches</h2>
    <div class="grid">
      {benign_cards if benign_res else '<div class="empty">No benign images in index.</div>'}
    </div>
  </div>
  <div class="section">
    <h2><span>🔴</span>Top {len(malignant_res)} Malignant Matches</h2>
    <div class="grid">
      {malignant_cards if malignant_res else '<div class="empty">No malignant images in index.</div>'}
    </div>
  </div>
</div>

<script>
  function openModal(matchB64, queryB64, label, sim, dist, source, fname) {{
    document.getElementById('modal-query').src = 'data:image/jpeg;base64,' + queryB64;
    document.getElementById('modal-match').src = 'data:image/jpeg;base64,' + matchB64;
    document.getElementById('modal-label-val').textContent  = label;
    document.getElementById('modal-label-span').textContent = label;
    document.getElementById('modal-sim').textContent    = sim;
    document.getElementById('modal-dist').textContent   = dist;
    document.getElementById('modal-source').textContent = source;
    document.getElementById('modal-fname').textContent  = fname;

    const block = document.getElementById('modal-label-block');
    block.className = 'stat-block ' +
      (label.toLowerCase() === 'benign' ? 'hl-benign' : 'hl-malignant');
    document.getElementById('modal-label-span').style.color =
      label.toLowerCase() === 'benign' ? '#00e5a0' : '#ff4060';

    document.getElementById('cbir-modal').classList.add('open');
    document.body.style.overflow = 'hidden';
  }}

  function closeModal() {{
    document.getElementById('cbir-modal').classList.remove('open');
    document.body.style.overflow = '';
  }}

  document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeModal(); }});
</script>
"""
        return html

    # ── State ─────────────────────────────────────────────────────────────────
    _state = {"query_img": None, "benign": [], "malignant": []}

    def run_search(uploaded_img, k_benign, k_malignant):
        if uploaded_img is None:
            return "<div style='color:#6b7592;padding:32px;font-family:monospace'>Upload an image to begin.</div>"

        img = Image.fromarray(uploaded_img).convert("RGB")
        _state["query_img"] = img

        benign_res, malignant_res = query_image_split(
            img, index, extractor,
            top_k_benign=int(k_benign),
            top_k_malignant=int(k_malignant),
        )
        _state["benign"]    = benign_res
        _state["malignant"] = malignant_res

        return results_to_html(benign_res, malignant_res, img)

    # ── UI ────────────────────────────────────────────────────────────────────
    css = """
    .gradio-container { background: #080a0f !important; padding-top: 64px !important; }
    footer { display: none !important; }
    #title-md h1 {
      font-family: 'Syne', sans-serif !important;
      font-size: 26px !important;
      color: #e8ecf5 !important;
      margin-bottom: 2px !important;
    }
    #title-md p {
      color: #6b7592 !important;
      font-size: 13px !important;
      font-family: 'DM Mono', monospace !important;
    }

    /* ── OralGuard Nav Bar ── */
    #oralguard-nav {
      position: fixed; top: 0; left: 0; right: 0; z-index: 9000;
      background: rgba(15, 11, 6, 0.97); backdrop-filter: blur(12px);
      border-bottom: 1px solid #2a2520;
      padding: 0 40px; display: flex; align-items: center; justify-content: space-between;
      height: 64px; font-family: 'Source Sans 3', sans-serif;
    }
    #oralguard-nav .og-logo {
      display: flex; align-items: center; gap: 10px; text-decoration: none;
    }
    #oralguard-nav .og-mark {
      width: 32px; height: 32px; background: #b84c2a; border-radius: 8px;
      display: flex; align-items: center; justify-content: center;
      font-size: 16px; color: white; font-weight: bold; font-family: serif;
    }
    #oralguard-nav .og-wordmark {
      font-family: 'Playfair Display', Georgia, serif; font-size: 20px; color: #f0ece6;
    }
    #oralguard-nav .og-wordmark span { color: #d4623a; }
    #oralguard-nav .og-links { display: flex; align-items: center; gap: 6px; }
    #oralguard-nav .og-link {
      padding: 7px 14px; font-size: 13px; color: #7a6e63;
      border-radius: 4px; border: 1px solid transparent;
      background: none; transition: all 0.15s;
      text-decoration: none; display: inline-block;
    }
    #oralguard-nav .og-link:hover { color: #f0ece6; border-color: #3a3530; background: #1a1510; }
    #oralguard-nav .og-link.active { color: #d4623a; border-color: rgba(184,76,42,0.35); background: rgba(184,76,42,0.12); font-weight: 600; }
    """

    nav_html = """
    <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=Playfair+Display:wght@400;700&family=Source+Sans+3:wght@300;400;600&display=swap" rel="stylesheet">
    <div id="oralguard-nav">
      <a class="og-logo" href="index.html">
        <div class="og-mark">+</div>
        <div class="og-wordmark">Oral<span>Guard</span></div>
      </a>
      <div class="og-links">
        <a class="og-link" href="index.html">Home</a>
        <a class="og-link" href="Questionnaire.html">Risk Screener</a>
        <a class="og-link active" href="#">Image Matcher</a>
      </div>
    </div>
    """

    with gr.Blocks(
        title="Oral Cancer CBIR",
        css=css,
        head=nav_html,
    ) as demo:

        gr.Markdown(
            """# 🔬 Oral Cancer CBIR
            Upload an oral lesion image to retrieve the most visually similar benign and malignant images from the database. Click any result to compare side-by-side.

            > ⚠️ **Research/educational use only — not a clinical diagnostic tool.**
            """,
            elem_id="title-md",
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=240):
                img_input   = gr.Image(label="Upload Lesion Image", type="numpy", height=260)
                k_benign    = gr.Slider(1, 20, value=top_k_benign,    step=1, label="Top-K Benign")
                k_malignant = gr.Slider(1, 20, value=top_k_malignant, step=1, label="Top-K Malignant")
                run_btn     = gr.Button("🔍  Search", variant="primary", size="lg")

            with gr.Column(scale=3):
                results_html = gr.HTML(
                    value="<div style='color:#6b7592;padding:40px 24px;"
                          "font-family:DM Mono,monospace;font-size:13px'>"
                          "Upload an image and click Search to begin.</div>"
                )

        run_btn.click(
            fn=run_search,
            inputs=[img_input, k_benign, k_malignant],
            outputs=[results_html],
        )
        img_input.change(
            fn=run_search,
            inputs=[img_input, k_benign, k_malignant],
            outputs=[results_html],
        )

    demo.launch(share=False, server_name=host, server_port=port)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Launch the Oral Cancer CBIR Gradio web interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--top_k_benign", type=int, default=5,
        help="Default number of top benign matches to show (default 5).",
    )
    parser.add_argument(
        "--top_k_malignant", type=int, default=5,
        help="Default number of top malignant matches to show (default 5).",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
        help="Compute device for embedding extraction (default: cpu).",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host address to bind to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to run the server on (default: 7860).",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA not available, falling back to CPU.")
        args.device = "cpu"

    print(f"\n{'─'*50}")
    print(f"  Oral Cancer CBIR — Web Application")
    print(f"  device={args.device}  index={INDEX_PATH}")
    print(f"{'─'*50}")

    print("\n  Loading feature extractor...")
    extractor = MobileNetV2Extractor(device=args.device)
    print("  ✓ Ready")

    print("\n  Loading search index...")
    index = load_index()

    print(f"\n  Launching at http://{args.host}:{args.port}\n")
    launch_app(
        extractor, index,
        top_k_benign=args.top_k_benign,
        top_k_malignant=args.top_k_malignant,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()