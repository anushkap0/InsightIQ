import sys, platform
import os
import warnings
    try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass
import pandas as pd

import torch
import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline
from dotenv import load_dotenv
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.rag_chatbot import RAGChatbot
from utils.trend_detector import TrendDetector

load_dotenv()
warnings.filterwarnings("ignore")

# ── MUST be the very first Streamlit call ──────────────────────────────────────
st.set_page_config(
    page_title="InsightIQ: AI-Powered Business Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ── CSS (one block, no duplicates) ─────────────────────────────────────────────
st.markdown("""
<style>
/* Animated gradient background */
.stApp {
    background: linear-gradient(315deg, #0f172a 3%, #1e293b 38%, #0f766e 68%, #312e81 98%);
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
    background-attachment: fixed;
}
@keyframes gradientMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.stApp > header { background-color: transparent; }
.block-container  { padding-top: 2rem; padding-bottom: 2rem; }

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 18px;
    padding: 1.2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(2,6,23,0.75);
    backdrop-filter: blur(10px);
}

/* Typography — everything white on the dark background */
h1, h2, h3, h4, h5, h6,
p, label, span, li, td, th,
.stMarkdown, .stMarkdown p,
div[data-testid="stText"] { color: white !important; }

.main-header    { font-size:2.5rem; font-weight:700; color:white !important; margin-bottom:1rem; }
.section-header { font-size:1.8rem; font-weight:600; color:white !important; margin:1.5rem 0 1rem 0; }

/* Info / alert boxes */
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] li,
div[data-testid="stAlert"] td,
div[data-testid="stAlert"] th { color: white !important; }

/* Table inside st.info */
.stAlert table, .stAlert th, .stAlert td { color: white !important; }

/* Buttons */
button {
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(59,130,246,0.35);
}

/* Animated progress bar */
.progress-wrap {
    width:100%; height:10px; border-radius:999px;
    background:rgba(255,255,255,0.18); overflow:hidden;
    box-shadow:inset 0 2px 8px rgba(0,0,0,0.15);
}
.progress-fill {
    height:100%; width:40%;
    background:linear-gradient(90deg,#3b82f6,#22d3ee,#a855f7);
    border-radius:999px;
    animation:moveFill 2s ease-in-out infinite;
    box-shadow:0 0 12px rgba(59,130,246,0.6);
}
@keyframes moveFill {
    0%   { transform: translateX(-100%); }
    50%  { transform: translateX(0%); }
    100% { transform: translateX(100%); }
}
</style>
""", unsafe_allow_html=True)

# ── Session-state defaults ─────────────────────────────────────────────────────
if "analyzer"         not in st.session_state:
    st.session_state.analyzer         = SentimentAnalyzer()
if "chatbot"          not in st.session_state:
    st.session_state.chatbot          = None
if "df"               not in st.session_state:
    st.session_state.df               = None
if "sentiment_results" not in st.session_state:
    st.session_state.sentiment_results = None
if "analysis_done"    not in st.session_state:
    st.session_state.analysis_done    = False
if "analysis_df"      not in st.session_state:
    st.session_state.analysis_df      = None

# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource
def get_analyzer(model_name: str) -> SentimentAnalyzer:
    return SentimentAnalyzer(model_name=model_name)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📊 InsightIQ</div>', unsafe_allow_html=True)
st.markdown("AI-Powered Business Intelligence Platform")
st.divider()
if "analyzer" not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer(model_name="distilbert-base-uncased-finetuned-sst-2-english")
# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔧 Controls")

    uploaded_file = st.file_uploader(
        "Upload Customer Reviews (CSV)",
        type=["csv"],
        help="CSV with a review column like review_text, review, text, comment, or feedback"
    )

    if uploaded_file is not None:
        uploaded_file.seek(0)
        df_raw = pd.read_csv(uploaded_file)
        df_raw.columns = df_raw.columns.str.strip()

        # Normalise text column name
        col_map = {"Text": "review_text", "review": "review_text",
                   "text": "review_text", "Summary": "review_text",
                   "comment": "review_text", "feedback": "review_text"}
        if "review_text" not in df_raw.columns:
            for src, tgt in col_map.items():
                if src in df_raw.columns:
                    df_raw = df_raw.rename(columns={src: tgt})
                    break

        st.session_state.df = df_raw
        st.success(f"Loaded {len(df_raw):,} reviews!")

    st.divider()
    

    st.divider()
    st.subheader("⚡ Performance")
    batch_size = st.select_slider(
        "Batch size (larger = faster on GPU):",
        options=[16, 32, 64, 128, 256],
        value=64,
    )
    use_sample = st.checkbox("Use random sample (faster preview)", value=True)
    sample_size = None
    if use_sample:
        sample_size = st.number_input(
            "Sample size", min_value=100, max_value=500_000,
            value=10_000, step=1_000
        )

    st.divider()
    st.info("""
    **How to use:**
    1. Upload your CSV reviews
    2. Select AI model
    3. Click Analyze Sentiment
    4. View the dashboard
    5. Ask the AI chatbot
    """)

# ── Main content ───────────────────────────────────────────────────────────────
if st.session_state.df is None:
    #  Getting-started screen ───────────────────────────────────
    st.markdown('<div class="section-header">🎯 Get Started</div>', unsafe_allow_html=True)
    st.info("""
    **Upload a CSV file with customer reviews to analyze sentiment.**

    Your CSV should have at least one column with review text. Example:

    | review_text | date | customer_name |
    |---|---|---|
    | "Great food, excellent service!" | 2026-01-15 | Anushka |
    | "Delivery was too slow, food cold" | 2026-01-16 | Rajesh |
    """)

    if st.button("📝 Generate Demo Data for Testing"):
        st.session_state.df = pd.DataFrame({
            "review_text": [
                "Amazing food! Best restaurant in Moradabad. Staff was very friendly.",
                "Delivery took 2 hours, food was cold. Very disappointed.",
                "Great taste, reasonable price. Will definitely order again.",
                "Terrible experience. Food overpriced and quality was not good.",
                "Excellent vegetarian options. Clean and comfortable.",
                "Waited 45 minutes. Food was partially cold. Not happy.",
                "Best place for breakfast. Fresh ingredients and amazing flavors!",
                "Parking is difficult, but food is worth it. Recommend the kebabs.",
                "Water was not clean but food was good.",
                "Perfect for family dinner. Kids loved the pasta.",
                "Overpriced for the quantity. Taste was average.",
                "Fast delivery, hot food, great packaging. My new favorite!",
                "Swiggy order delayed but restaurant gave free dessert.",
                "Noodles too spicy. Requested less but they ignored.",
                "Wonderful! Clean restaurant, good music, delicious food."
            ],
            "date": [f"2026-01-{i:02d}" for i in range(15, 30)],
            "customer_name": [
                "Anushka","Rajesh","Priya","Vikram","Sneha",
                "Amit","Kavita","Mohit","Deepika","Suresh",
                "Nisha","Rahul","Pooja","Tarun","Meera"
            ]
        })
        st.success("Demo data loaded!")
        st.rerun()

else:
    # ─ analysis + dashboard ─────────────────────────────────────
    df = st.session_state.df

    st.success(f"Dataset loaded with **{len(df):,}** reviews")

    analyze_btn = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True
    )

    if analyze_btn:
        if "review_text" not in df.columns:
            st.error(f"Missing 'review_text' column. Found: {list(df.columns)}")
            st.stop()

        texts = df["review_text"].fillna("").astype(str).tolist()

        # Optional sampling
        import random
        if use_sample and sample_size and sample_size < len(texts):
            random.seed(42)
            indices  = random.sample(range(len(texts)), int(sample_size))
            texts    = [texts[i] for i in indices]
            work_df  = df.iloc[indices].copy().reset_index(drop=True)
        else:
            work_df = df.copy()

        progress_bar = st.progress(0)
        status_text  = st.empty()
        status_text.markdown("""
        <div class="progress-wrap"><div class="progress-fill"></div></div>
        <p style="color:white;margin-top:.5rem;">Analyzing reviews with HuggingFace AI…</p>
        """, unsafe_allow_html=True)

        def on_progress(processed: int, total: int):
            progress_bar.progress(processed / total)
            status_text.text(f"Processed {processed:,} / {total:,} reviews ({processed/total:.0%})")

        with st.spinner(""):
            results = st.session_state.analyzer.analyze_batch(
                texts,
                batch_size=batch_size,
                progress_callback=on_progress,
            )

        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")

        work_df["sentiment"]   = results["sentiments"]
        work_df["confidence"]  = results["confidences"]

        st.session_state.analysis_df   = work_df
        st.session_state.analysis_done = True

    # ── Dashboard (shown after analysis) ──────────────────────────────────────
    if st.session_state.analysis_done and st.session_state.analysis_df is not None:
        df_result = st.session_state.analysis_df

        st.divider()
        st.markdown('<div class="section-header">📈 Sentiment Dashboard</div>', unsafe_allow_html=True)

        sentiment_counts = df_result["sentiment"].value_counts()
        total_reviews    = len(df_result)
        positive_pct     = sentiment_counts.get("POSITIVE", 0) / total_reviews * 100
        negative_pct     = sentiment_counts.get("NEGATIVE", 0) / total_reviews * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Total Reviews",  f"{total_reviews:,}", delta="Analysis done")
        col2.metric("😊 Positive",       f"{positive_pct:.1f}%",
                    delta=f"{sentiment_counts.get('POSITIVE',0):,} reviews")
        col3.metric("😠 Negative",       f"{negative_pct:.1f}%",
                    delta=f"{sentiment_counts.get('NEGATIVE',0):,} reviews",
                    delta_color="inverse")

        st.divider()

        # Charts
        labels = ["Positive", "Negative", "Neutral"]
        sizes  = [
            sentiment_counts.get("POSITIVE", 0),
            sentiment_counts.get("NEGATIVE", 0),
            sentiment_counts.get("NEUTRAL",  0),
        ]
        colors = ["#10B981", "#EF4444", "#F59E0B"]

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            fig1.patch.set_alpha(0)
            ax1.set_facecolor("none")
            ax1.pie(sizes, labels=labels, colors=colors,
                    autopct="%1.1f%%", startangle=90)
            ax1.set_title("Sentiment Distribution", fontsize=14,
                          fontweight="bold", color="white")
            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            fig2.patch.set_alpha(0)
            ax2.set_facecolor("none")
            ax2.bar(labels, sizes, color=colors, edgecolor="white", alpha=0.85)
            ax2.set_title("Sentiment Counts", fontsize=14,
                          fontweight="bold", color="white")
            ax2.set_ylabel("Number of Reviews", fontsize=12, color="white")
            ax2.tick_params(colors="white")
            ax2.set_ylim(0, max(sizes) + 1 if max(sizes) > 0 else 1)
            for i, v in enumerate(sizes):
                ax2.text(i, v + 0.1, str(v), ha="center",
                         fontweight="bold", color="white")
            st.pyplot(fig2)
            plt.close(fig2)

        st.divider()

        # ── Review table ───────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📝 Individual Reviews</div>',
                    unsafe_allow_html=True)

        review_cols = ["review_text"]
        for c in ("date", "customer_name"):
            if c in df_result.columns:
                review_cols.append(c)
        review_cols += ["sentiment", "confidence"]

        display_df = df_result[review_cols].rename(columns={
            "review_text":   "Review Text",
            "date":          "Date",
            "customer_name": "Customer",
            "sentiment":     "Sentiment",
            "confidence":    "Confidence",
        })

        def color_sentiment(val):
            if val == "POSITIVE":
                return "background-color:#D1FAE5; color:#065F46"
            elif val == "NEGATIVE":
                return "background-color:#FEE2E2; color:#991B1B"
            return "background-color:#FEF3C7; color:#92400E"

        st.dataframe(
            display_df.style.map(color_sentiment, subset=["Sentiment"]),
            use_container_width=True, height=400
        )

        st.divider()

        # ── Trend detection ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Trend Detection</div>',
                    unsafe_allow_html=True)

        trend_detector = TrendDetector()
        trends = trend_detector.detect_trends(df_result)

        if trends:
            st.success(f"🔔 Found {len(trends)} emerging trends!")
            for i, trend in enumerate(trends[:3], 1):
                with st.expander(f"Trend #{i}: {trend['topic']}", expanded=True):
                    st.write(f"**Description:** {trend['description']}")
                    st.write(f"**Direction:** {trend['direction']}")
                    st.write(f"**Confidence:** {trend['confidence']:.2f}")
                    st.write(f"**Keywords:** {', '.join(trend['keywords'])}")
        else:
            st.info("No significant trends detected in this dataset.")

        st.divider()

        # ── RAG Chatbot ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">💬 AI Insights Chatbot</div>',
                    unsafe_allow_html=True)
        st.info("""
        Ask questions like:
        - "What do customers say about delivery speed?"
        - "What are the main complaints?"
        - "Show me positive feedback about pricing"
        """)

        if st.session_state.chatbot is None:
            with st.spinner("Loading AI chatbot…"):
                st.session_state.chatbot = RAGChatbot()
                st.session_state.chatbot.load_documents(
                    df_result["review_text"].tolist()
                )

        for msg in st.session_state.chatbot.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_query = st.chat_input("Ask about customer reviews…")
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            with st.spinner("AI is analysing…"):
                response = st.session_state.chatbot.get_response(user_query)
            with st.chat_message("assistant"):
                st.write(response)
            st.session_state.chatbot.chat_history.append(
                {"role": "user",      "content": user_query}
            )
            st.session_state.chatbot.chat_history.append(
                {"role": "assistant", "content": response}
            )

        st.divider()

        # ── Recommendations ────────────────────────────────────────────────────
        st.markdown('<div class="section-header">💡 Business Recommendations</div>',
                    unsafe_allow_html=True)

        recommendations = trend_detector.generate_recommendations(df_result)
        if recommendations:
            for rec in recommendations[:3]:
                st.markdown(f"""
                <div class="glass-card" style="margin:0.5rem 0;">
                    <strong>{rec['category']}</strong><br>{rec['recommendation']}
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No major recommendation issues detected.")

        st.divider()

        # ── Export ─────────────────────────────────────────────────────────────
        csv_bytes = df_result.to_csv(index=False).encode()
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_bytes,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#9CA3AF; font-size:0.9rem;">
    Built with HuggingFace AI &bull; Streamlit &bull; LangChain &bull; RAG
</div>
""", unsafe_allow_html=True)
