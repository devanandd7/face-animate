"""
Streamlit Web Interface — AI Face Animator
==========================================
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import os
import tempfile
from animate import init_gemini, animate_image

st.set_page_config(
    page_title="AI Face Animator",
    page_icon="🎭",
    layout="centered"
)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get your key from https://aistudio.google.com/apikey"
    )
    if api_key:
        try:
            init_gemini(api_key)
            st.success("Gemini ✅ Connected")
        except Exception as e:
            st.error(f"Gemini ❌ {e}")

    st.divider()
    st.markdown("""
### 🎬 Animations
| Command | Effect |
|---|---|
| `blink` | Both eyes blink |
| `wink left` | Left eye wink |
| `wink right` | Right eye wink |
| `smile` | Big smile 😊 |
| `eyebrow raise` | Surprise 🤨 |
| `nod` | Head nod |
| _Any sentence_ | Lip-sync speech |

### 💬 Examples
- *"Good morning everyone"*
- *"Hello how are you"*
- *"Make him blink twice"*
- *"She is smiling"*
- *"Namaste"*
""")

# ── Main ──────────────────────────────────────────────────────────
st.title("🎭 AI Face Animator")
st.caption("Upload a face photo → type what it should do → get an animated GIF")

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader(
        "📤 Upload face image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Clear, front-facing photo works best"
    )
with col2:
    user_text = st.text_input(
        "💬 What should the face do?",
        placeholder="e.g., Good morning!  /  Make her blink",
    )

st.divider()

if uploaded and user_text:
    if st.button("✨ Animate!", type="primary", use_container_width=True):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
        else:
            with st.spinner("🎬 Generating animation…"):
                suffix = os.path.splitext(uploaded.name)[1] or ".jpg"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
                    tmp_in.write(uploaded.read())
                    tmp_in_path = tmp_in.name

                tmp_out_path = tempfile.mktemp(suffix=".gif")

                try:
                    result = animate_image(
                        image_path    = tmp_in_path,
                        user_text     = user_text,
                        output_path   = tmp_out_path,
                        gemini_api_key= api_key,
                    )

                    if result and os.path.exists(tmp_out_path):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.subheader("📷 Original")
                            st.image(tmp_in_path, use_container_width=True)
                        with c2:
                            st.subheader("🎬 Animated")
                            st.image(tmp_out_path, use_container_width=True)

                        with open(tmp_out_path, "rb") as f:
                            gif_bytes = f.read()

                        st.download_button(
                            "⬇️ Download GIF",
                            data      = gif_bytes,
                            file_name = "animation.gif",
                            mime      = "image/gif",
                            use_container_width=True,
                        )
                        kb = len(gif_bytes) / 1024
                        st.success(f"✅ Done! ({kb:.1f} KB, check preview above)")
                    else:
                        st.error("❌ No face detected. Try a clearer front-facing photo.")

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback; traceback.print_exc()

                finally:
                    for p in (tmp_in_path, tmp_out_path):
                        try: os.unlink(p)
                        except: pass

elif uploaded and not user_text:
    st.info("👆 Type what you want the face to do.")
elif user_text and not uploaded:
    st.info("👆 Upload a face image first.")
else:
    st.markdown(
        '<div style="text-align:center;color:#888;padding:40px">'
        '<h3>Upload a face image & type an instruction to begin</h3>'
        '</div>',
        unsafe_allow_html=True
    )
