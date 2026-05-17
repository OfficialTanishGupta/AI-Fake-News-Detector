import streamlit as st
import torch
import torch.nn as nn
import joblib
import time

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens · AI News Verifier",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0a0c10;
    --surface:   #111318;
    --border:    #1e2330;
    --accent:    #4f9cf9;
    --accent2:   #7b5ea7;
    --real:      #22c55e;
    --fake:      #ef4444;
    --muted:     #6b7280;
    --text:      #e8eaf0;
    --text-dim:  #9ca3af;
}

/* ── Base reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(79,156,249,0.10) 0%, transparent 60%),
        var(--bg) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    background: rgba(79,156,249,0.10);
    border: 1px solid rgba(79,156,249,0.25);
    border-radius: 100px;
    padding: 0.3rem 1rem;
    margin-bottom: 1.25rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin: 0 0 1rem;
    background: linear-gradient(135deg, #e8eaf0 30%, #4f9cf9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-dim);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── Card ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
}
.card-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.6rem;
}

/* ── Textarea override ── */
textarea {
    background: #13161d !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    resize: vertical !important;
    transition: border-color 0.2s !important;
}
textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,156,249,0.10) !important;
    outline: none !important;
}

/* ── Analyze button ── */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #4f9cf9, #7b5ea7) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.75rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Result banner ── */
.result-banner {
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
    display: flex;
    align-items: center;
    gap: 1.25rem;
}
.result-banner.real {
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.30);
}
.result-banner.fake {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.30);
}
.result-icon { font-size: 2.4rem; line-height: 1; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.01em;
}
.result-banner.real  .result-label { color: var(--real); }
.result-banner.fake  .result-label { color: var(--fake); }
.result-sub {
    font-size: 0.88rem;
    color: var(--text-dim);
    margin-top: 0.15rem;
}

/* ── Confidence bar ── */
.conf-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    font-size: 0.85rem;
    color: var(--text-dim);
}
.conf-track {
    height: 8px;
    background: var(--border);
    border-radius: 100px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s cubic-bezier(.22,1,.36,1);
}
.conf-fill.real { background: linear-gradient(90deg, #22c55e, #4ade80); }
.conf-fill.fake { background: linear-gradient(90deg, #ef4444, #f87171); }

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.25rem;
}
.stat-box {
    flex: 1;
    background: #13161d;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
}
.stat-lbl {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-top: 0.2rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2.5rem 0 1rem;
    font-size: 0.8rem;
    color: var(--muted);
    letter-spacing: 0.04em;
}
.footer a { color: var(--accent); text-decoration: none; }

/* ── Warning ── */
[data-testid="stAlert"] {
    background: rgba(251,191,36,0.08) !important;
    border: 1px solid rgba(251,191,36,0.25) !important;
    border-radius: 12px !important;
    color: #fbbf24 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] > div { color: var(--accent) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🔍 Powered by Deep Learning</div>
    <h1 class="hero-title">TruthLens</h1>
    <p class="hero-sub">
        Paste any news article below and our neural network will assess its
        credibility in seconds.
    </p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)


# ── Load vocab ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    vocab = joblib.load("../models/vocab.pkl")

    class FakeNewsClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.fc1      = nn.Linear(embed_dim, 256)
            self.relu1    = nn.ReLU()
            self.dropout  = nn.Dropout(0.3)
            self.fc2      = nn.Linear(256, 128)
            self.relu2    = nn.ReLU()
            self.fc3      = nn.Linear(128, 1)
            self.sigmoid  = nn.Sigmoid()

        def forward(self, x):
            x = self.embedding(x).mean(dim=1)
            x = self.dropout(self.relu1(self.fc1(x)))
            x = self.relu2(self.fc2(x))
            return self.sigmoid(self.fc3(x))

    vocab_size = len(vocab) + 1
    model = FakeNewsClassifier(vocab_size, embed_dim=128)
    model.load_state_dict(torch.load("../models/fake_news_dl_model.pt", weights_only=True))
    model.eval()
    return vocab, model

vocab, model = load_assets()


# ── Helpers ───────────────────────────────────────────────────────────────────
MAX_LEN = 200

def encode(text: str) -> list[int]:
    tokens  = text.lower().split()
    encoded = [vocab.get(w, 0) for w in tokens][:MAX_LEN]
    encoded += [0] * (MAX_LEN - len(encoded))
    return encoded

def word_count(text: str) -> int:
    return len(text.split())

def char_count(text: str) -> int:
    return len(text)


# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="card-label">📄 Article Text</div>', unsafe_allow_html=True)
news_input = st.text_area(
    label="",
    placeholder="Paste or type your news article here…",
    height=220,
    label_visibility="collapsed",
)

col_btn, _ = st.columns([1, 0.001])
with col_btn:
    analyze = st.button("Analyze Article →", use_container_width=True)


# ── Analysis ──────────────────────────────────────────────────────────────────
if analyze:
    stripped = news_input.strip()
    if not stripped:
        st.warning("⚠️  Please paste an article before analyzing.")
    else:
        with st.spinner("Running neural inference…"):
            time.sleep(0.4)          # brief dramatic pause
            encoded      = encode(stripped)
            input_tensor = torch.tensor([encoded], dtype=torch.long)
            with torch.no_grad():
                confidence = model(input_tensor).item()

        is_real   = confidence >= 0.5
        pct_real  = confidence * 100
        pct_fake  = (1 - confidence) * 100
        display_conf = pct_real if is_real else pct_fake
        verdict   = "REAL"    if is_real else "FAKE"
        cls       = "real"    if is_real else "fake"
        icon      = "✅"       if is_real else "🚨"
        sub_msg   = (
            "High probability this article contains accurate information."
            if is_real else
            "This article shows patterns commonly associated with misinformation."
        )

        # Result banner
        st.markdown(f"""
        <div class="result-banner {cls}">
            <div class="result-icon">{icon}</div>
            <div>
                <div class="result-label">{verdict} NEWS</div>
                <div class="result-sub">{sub_msg}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown(f"""
        <div class="card">
            <div class="card-label">Confidence Score</div>
            <div class="conf-row">
                <span>{verdict} probability</span>
                <strong style="color:var(--text)">{display_conf:.1f}%</strong>
            </div>
            <div class="conf-track">
                <div class="conf-fill {cls}" style="width:{display_conf}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stats row
        wc   = word_count(stripped)
        cc   = char_count(stripped)
        cov  = min(wc, MAX_LEN)
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-box">
                <div class="stat-val">{wc:,}</div>
                <div class="stat-lbl">Words</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{cc:,}</div>
                <div class="stat-lbl">Characters</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{cov}</div>
                <div class="stat-lbl">Tokens used</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{display_conf:.0f}%</div>
                <div class="stat-lbl">Confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with <strong>PyTorch</strong> + <strong>Streamlit</strong> &nbsp;·&nbsp;
    TruthLens v1.0
</div>
""", unsafe_allow_html=True)