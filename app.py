import streamlit as st
from rag.pipeline import ask_question, ask_question_stream


# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="AI Handbook | RAG Knowledge Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# Vintage Cream & Black Aesthetic Custom CSS
# ==========================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;0,800;1,600&family=Space+Grotesk:wght@400;500;600;700&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

    /* ==========================================
       GLOBAL ROOT & APP CANVAS
       ========================================== */
    html, body, .stApp, [data-testid="stAppViewContainer"], section.main, [data-testid="stAppViewBlockContainer"], .main {
        background-color: #F8F5EE !important;
        background: #F8F5EE !important;
        color: #111111 !important;
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
        overflow-anchor: none !important;
        scroll-behavior: auto !important;
    }

    /* ==========================================
       STREAMLIT HEADER & TOOLBAR
       ========================================== */
    header[data-testid="stHeader"], [data-testid="stHeader"], .stApp > header {
        background-color: transparent !important;
        background: transparent !important;
        pointer-events: none !important;
        z-index: 99 !important;
    }

    [data-testid="stToolbar"], .stAppToolbar {
        background-color: transparent !important;
        background: transparent !important;
        pointer-events: none !important;
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    #MainMenu,
    [data-testid="stHeaderActionElements"],
    [data-testid="stToolbarActionButton"],
    [data-testid="stAppDeployButton"],
    .stAppDeployButton,
    [data-testid="manage-app-button"] {
        display: none !important;
    }

    /* ==========================================
       SIDEBAR TOGGLE BUTTON
       ========================================== */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"],
    button[data-testid="baseButton-headerNoPadding"] {
        pointer-events: auto !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 100000 !important;
        background-color: #FFFDF8 !important;
        border: 1.5px solid #111111 !important;
        box-shadow: 2.5px 2.5px 0px #111111 !important;
        border-radius: 6px !important;
        color: #111111 !important;
        padding: 0.35rem 0.6rem !important;
        margin: 0.4rem !important;
        cursor: pointer !important;
        transition: all 0.12s ease !important;
    }

    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stExpandSidebarButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"]:hover,
    [data-testid="collapsedControl"]:hover,
    button[data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="stSidebarCollapseButton"] button:hover,
    button[data-testid="baseButton-headerNoPadding"]:hover {
        background-color: #111111 !important;
        color: #F8F5EE !important;
        transform: translate(1px, 1px) !important;
        box-shadow: 1.5px 1.5px 0px #111111 !important;
    }

    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="stSidebarCollapsedControl"] *,
    [data-testid="collapsedControl"] * {
        color: inherit !important;
        fill: currentColor !important;
        stroke: currentColor !important;
    }

    /* ==========================================
       FIXED BOTTOM CHAT AREA
       ========================================== */
    [data-testid="stBottom"], [data-testid="stBottom"] > div, .stChatFloatingInputContainer, footer {
        background-color: #F8F5EE !important;
        background: #F8F5EE !important;
        color: #111111 !important;
    }

    /* ==========================================
       MAIN LAYOUT CONTAINER
       ========================================== */
    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 6rem !important;
        max-width: 860px;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* ==========================================
       SIDEBAR STYLING
       ========================================== */
    section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #F1ECE0 !important;
        border-right: 1.5px solid #111111 !important;
        color: #111111;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] .block-container {
        padding: 1.25rem 1.1rem !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label {
        color: #111111;
    }

    /* ==========================================
       SIDEBAR CARDS
       ========================================== */
    .sidebar-card {
        background-color: #FFFDF8 !important;
        border: 1.5px solid #111111 !important;
        box-shadow: 2.5px 2.5px 0px #111111 !important;
        border-radius: 6px !important;
        padding: 0.75rem 0.85rem !important;
        margin-bottom: 0.75rem !important;
        box-sizing: border-box !important;
        width: 100% !important;
    }

    .sidebar-title-card {
        text-align: center;
        padding: 0.85rem 0.9rem !important;
        margin-bottom: 0.85rem !important;
    }

    .sidebar-brand-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #111111;
        line-height: 1.2;
    }

    .sidebar-brand-sub {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #555555;
        margin-top: 0.25rem;
    }

    .sidebar-section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #111111;
        margin-top: 0.85rem;
        margin-bottom: 0.45rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    .sidebar-history-item {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem;
        padding: 0.45rem 0.6rem;
        background-color: #FFFDF8;
        border: 1.5px solid #111111;
        box-shadow: 2px 2px 0px #111111;
        margin-bottom: 0.45rem;
        border-radius: 5px;
        color: #111111;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        box-sizing: border-box;
    }

    .sidebar-status-card {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        line-height: 1.6;
    }

    .status-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.15rem 0;
        border-bottom: 1px dashed #E5DFD1;
    }

    .status-row:last-child {
        border-bottom: none;
    }

    .sidebar-about-card {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.82rem;
        line-height: 1.5;
        color: #333333;
    }

    /* ==========================================
       DIVIDERS
       ========================================== */
    hr {
        border-color: #111111 !important;
        opacity: 0.2;
        margin: 1rem 0 !important;
    }

    /* ==========================================
       MAIN HEADER
       ========================================== */
    .vintage-header {
        border: 2px solid #111111;
        background-color: #FFFDF8;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.3rem;
        box-shadow: 3px 3px 0px #111111;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.6rem;
        box-sizing: border-box;
        width: 100%;
        border-radius: 8px;
    }

    .vintage-header-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: #111111;
        margin: 0;
        line-height: 1.15;
    }

    .vintage-header-sub {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #444444;
        margin-top: 0.3rem;
    }

    .vintage-badge {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.35rem 0.75rem;
        background: #111111;
        color: #F8F5EE !important;
        border: 1px solid #111111;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border-radius: 4px;
    }

    /* ==========================================
       WELCOME HERO
       ========================================== */
    .vintage-welcome-card {
        background-color: #FFFDF8;
        border: 1.5px solid #111111;
        padding: 1.5rem 1.6rem 1.3rem 1.6rem;
        text-align: center;
        box-shadow: 3px 3px 0px #111111;
        margin-bottom: 1.2rem;
        border-radius: 8px;
        box-sizing: border-box;
        width: 100%;
    }

    .vintage-stamp {
        display: inline-block;
        border: 1px solid #111111;
        padding: 0.2rem 0.75rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        background: #111111;
        color: #F8F5EE !important;
        margin-bottom: 0.75rem;
        border-radius: 4px;
    }

    .vintage-welcome-card h2 {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.75rem;
        font-weight: 700;
        color: #111111 !important;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.01em;
    }

    .vintage-welcome-card p {
        font-family: 'Space Grotesk', sans-serif;
        color: #333333 !important;
        font-size: 0.95rem;
        line-height: 1.55;
        max-width: 580px;
        margin: 0 auto;
    }

    /* ==========================================
       SECTION TITLES
       ========================================== */
    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #111111;
        margin-top: 0.75rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ==========================================
       BUTTONS
       ========================================== */
    .stButton > button,
    .stButton > button[kind="primary"],
    .stButton > button[kind="secondary"] {
        background-color: #FFFDF8 !important;
        color: #111111 !important;
        border: 1.5px solid #111111 !important;
        border-radius: 6px !important;
        box-shadow: 2.5px 2.5px 0px #111111 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.55rem 0.85rem !important;
        transition: all 0.12s ease !important;
        text-align: center !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    .stButton > button *,
    .stButton > button p,
    .stButton > button span {
        color: #111111 !important;
    }

    .stButton > button:hover,
    .stButton > button[kind="primary"]:hover,
    .stButton > button[kind="secondary"]:hover {
        background-color: #111111 !important;
        color: #F8F5EE !important;
        transform: translate(1px, 1px) !important;
        box-shadow: 1.5px 1.5px 0px #111111 !important;
    }

    .stButton > button:hover *,
    .stButton > button:hover p,
    .stButton > button:hover span,
    .stButton > button:hover div {
        color: #F8F5EE !important;
    }

    /* ==========================================
       CHAT MESSAGES
       ========================================== */
    .stChatMessage {
        background-color: #FFFDF8 !important;
        border: 1.5px solid #111111 !important;
        border-radius: 8px !important;
        padding: 1.1rem 1.2rem 1.3rem 1.2rem !important;
        margin-bottom: 1.1rem !important;
        box-shadow: 3px 3px 0px #111111 !important;
        box-sizing: border-box !important;
    }

    .stChatMessage p,
    .stChatMessage li,
    .stChatMessage span,
    .stChatMessage strong {
        color: #111111 !important;
        line-height: 1.6 !important;
        font-size: 0.95rem !important;
    }

    .stChatMessage h1,
    .stChatMessage h2,
    .stChatMessage h3,
    .stChatMessage h4 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #111111 !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* ==========================================
       CHAT AVATARS
       ========================================== */
    div[data-testid="stChatMessageAvatarUser"],
    div[data-testid="stChatMessageAvatarAssistant"],
    .stChatMessageAvatar {
        background-color: #EFE9DB !important;
        border: 1px solid #111111 !important;
        border-radius: 6px !important;
        color: #111111 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.05rem !important;
        width: 2.1rem !important;
        height: 2.1rem !important;
        line-height: 1 !important;
        box-shadow: 1px 1px 0px #111111 !important;
    }

    div[data-testid="stChatMessageAvatarUser"] svg,
    div[data-testid="stChatMessageAvatarAssistant"] svg {
        fill: #111111 !important;
        stroke: #111111 !important;
    }

    /* ==========================================
       MESSAGE HEADERS
       ========================================== */
    .msg-header-user {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #444444;
        margin-bottom: 0.35rem;
    }

    .msg-header-ai {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px dashed #D6D0C2;
    }

    .msg-role-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: #111111;
        letter-spacing: 0.02em;
        line-height: 1;
    }

    /* ==========================================
       CHATGPT-STYLE 3-DOT LOADING ANIMATION
       ========================================== */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 0.5rem 0.2rem;
        min-height: 24px;
    }

    .typing-dot {
        width: 8px;
        height: 8px;
        background-color: #111111;
        border-radius: 50%;
        display: inline-block;
        animation: typingBounce 1.4s infinite ease-in-out both;
    }

    .typing-dot:nth-child(1) {
        animation-delay: -0.32s;
    }

    .typing-dot:nth-child(2) {
        animation-delay: -0.16s;
    }

    .typing-dot:nth-child(3) {
        animation-delay: 0s;
    }

    @keyframes typingBounce {
        0%, 80%, 100% {
            transform: scale(0.6);
            opacity: 0.3;
        }
        40% {
            transform: scale(1.15);
            opacity: 1;
        }
    }

    /* ==========================================
       RAG TAG
       ========================================== */
    .rag-tag {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.35rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #111111;
        background-color: #EFE9DB;
        border: 1px solid #111111;
        padding: 0.3rem 0.65rem;
        line-height: 1;
        border-radius: 4px;
        box-sizing: border-box;
    }

    /* ==========================================
       RAG SOURCES CARD
       ========================================== */
    .rag-sources-card {
        background-color: #F4EFE5;
        border: 1px dashed #C8C0AF;
        border-top: 1.5px dashed #111111;
        padding: 0.85rem 1rem;
        margin-top: 1rem;
        margin-bottom: 0.75rem !important;
        border-radius: 6px;
    }

    .rag-sources-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #111111;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
        line-height: 1.2;
    }

    .rag-source-item {
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        color: #2b2b2b;
        margin-top: 0.3rem;
        line-height: 1.5;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* ==========================================
       CHAT INPUT
       ========================================== */
    .stChatInputContainer {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        background-color: transparent !important;
    }

    div[data-testid="stChatInput"], .stChatInput {
        border: 2px solid #111111 !important;
        border-radius: 8px !important;
        background-color: #FFFDF8 !important;
        box-shadow: 3px 3px 0px #111111 !important;
        padding: 0.3rem 0.5rem !important;
        box-sizing: border-box !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: #111111 !important;
        box-shadow: 3px 3px 0px #111111 !important;
    }

    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] [data-baseweb="base-input"],
    div[data-testid="stChatInput"] [data-baseweb="textarea"] {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        border-radius: 6px !important;
    }

    div[data-testid="stChatInput"] textarea {
        color: #111111 !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.95rem !important;
        background-color: transparent !important;
        border: none !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #777777 !important;
        font-size: 0.92rem;
    }

    /* ==========================================
       CHAT INPUT SUBMIT BUTTON
       ========================================== */
    div[data-testid="stChatInput"] button,
    button[data-testid="stChatInputSubmitButton"],
    [data-testid="stChatInputSubmitButton"] {
        background-color: #111111 !important;
        color: #FFFDF8 !important;
        border-radius: 6px !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        transition: all 0.12s ease !important;
    }

    div[data-testid="stChatInput"] button:hover,
    button[data-testid="stChatInputSubmitButton"]:hover {
        background-color: #333333 !important;
    }

    div[data-testid="stChatInput"] button:focus,
    div[data-testid="stChatInput"] button:focus-visible,
    div[data-testid="stChatInput"] button:active,
    button[data-testid="stChatInputSubmitButton"]:focus,
    button[data-testid="stChatInputSubmitButton"]:focus-visible,
    button[data-testid="stChatInputSubmitButton"]:active {
        outline: none !important;
        border: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] button *,
    button[data-testid="stChatInputSubmitButton"] * {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] button svg path:not([fill="none"]),
    button[data-testid="stChatInputSubmitButton"] svg path:not([fill="none"]) {
        fill: #FFFDF8 !important;
        stroke: none !important;
    }

    /* ==========================================
       HIDE COMPONENT IFRAMES
       ========================================== */
    iframe[title="streamlit.components.v1.html"] {
        display: none !important;
        height: 0px !important;
        width: 0px !important;
        position: absolute !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# Session State Initialization
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history_queries" not in st.session_state:
    st.session_state.history_queries = []


# ==========================================
# Helper Functions
# ==========================================
def handle_query(query_text: str):
    if not query_text or not query_text.strip():
        return

    cleaned_query = query_text.strip()
    if cleaned_query not in st.session_state.history_queries:
        st.session_state.history_queries.append(cleaned_query)

    st.session_state.messages.append({
        "role": "user",
        "content": cleaned_query,
    })

    st.session_state.pending_prompt = cleaned_query


def clear_chat():
    st.session_state.messages = []
    st.session_state.history_queries = []
    if "pending_prompt" in st.session_state:
        del st.session_state.pending_prompt


def render_sources_card(sources):
    if not sources:
        return

    # Deduplicate source/page pairs while preserving order
    unique_sources = []
    seen = set()
    for src in sources:
        source_name = src.get("source", "Unknown document")
        page = src.get("page", "?")
        key = (source_name, str(page))
        if key not in seen:
            seen.add(key)
            unique_sources.append({"source": source_name, "page": page})

    if not unique_sources:
        return

    sources_html = '<div class="rag-sources-card"><div class="rag-sources-header">📄 SOURCES</div>'
    for src in unique_sources:
        sources_html += f'<div class="rag-source-item">📄 {src["source"]} — Page {src["page"]}</div>'
    sources_html += "</div>"

    st.markdown(sources_html, unsafe_allow_html=True)


# ==========================================
# Scroll Reset on Initial Page Load
# ==========================================
if not st.session_state.messages:
    st.html(
        """
        <script>
        function forceScrollTop() {
            try {
                const elements = [
                    document.querySelector('section[data-testid="stMain"]'),
                    document.querySelector('section.main'),
                    document.querySelector('.stAppViewContainer'),
                    document.querySelector('[data-testid="stAppViewContainer"]'),
                    document.querySelector('[data-testid="stAppViewBlockContainer"]'),
                    document.documentElement,
                    document.body
                ];
                elements.forEach(el => {
                    if (el && el.scrollTop !== undefined) {
                        el.scrollTop = 0;
                    }
                });
                window.scrollTo(0, 0);
            } catch(e) {}
        }
        forceScrollTop();
        window.addEventListener('load', forceScrollTop);
        window.addEventListener('focusin', forceScrollTop);
        let count = 0;
        const timer = setInterval(() => {
            forceScrollTop();
            count++;
            if (count > 25) {
                clearInterval(timer);
                window.removeEventListener('focusin', forceScrollTop);
            }
        }, 40);
        </script>
        """
    )


# ==========================================
# Sidebar
# ==========================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card sidebar-title-card">
            <div class="sidebar-brand-title">📖 AI Handbook</div>
            <div class="sidebar-brand-sub">RAG Retrieval Studio</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.button(
        "＋ New Chat",
        on_click=clear_chat,
        use_container_width=True,
    )

    st.divider()

    # About
    st.markdown('<div class="sidebar-section-header">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-card sidebar-about-card">
            RAG-powered conversational assistant for fast lookup and synthesis of official handbook guidelines, policies, and operational procedures.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Conversation History
    if st.session_state.history_queries:
        st.markdown('<div class="sidebar-section-header">🕘 Recent Questions</div>', unsafe_allow_html=True)
        for query in reversed(st.session_state.history_queries[-5:]):
            display_query = query if len(query) <= 55 else query[:52] + "..."
            st.markdown(f'<div class="sidebar-history-item">{display_query}</div>', unsafe_allow_html=True)

    # System Status
    st.markdown('<div class="sidebar-section-header">⚙️ System</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-card sidebar-status-card">
            <div class="status-row"><span>Retrieval</span><strong>ONLINE</strong></div>
            <div class="status-row"><span>LLM</span><strong>OLLAMA</strong></div>
            <div class="status-row"><span>Mode</span><strong>RAG</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# Main Layout
# ==========================================

# Top Header
st.markdown(
    """
    <div class="vintage-header">
        <div>
            <div class="vintage-header-title">📖 AI Handbook</div>
            <div class="vintage-header-sub">RAG-Powered Organizational Knowledge Base</div>
        </div>
        <div class="vintage-badge">Status: Active RAG</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Chat Input at bottom
chat_input_val = st.chat_input("Ask a question about the handbook, policies, or procedures...")
if chat_input_val:
    handle_query(chat_input_val)

# Welcome Screen (when no messages)
welcome_placeholder = st.empty()

if not st.session_state.messages:
    with welcome_placeholder.container():
        st.markdown(
            """
            <div class="vintage-welcome-card">
                <div class="vintage-stamp">AI Knowledge Assistant</div>
                <h2>Welcome to the AI Handbook</h2>
                <p>Your intelligent retrieval assistant for rapid inquiry into organizational guidelines, workplace policies, and procedural documentation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">💡 Suggested Inquiries:</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.button(
                "📋 What is the remote work policy?",
                on_click=handle_query,
                args=("What is the remote work policy?",),
                use_container_width=True,
            )
            st.button(
                "💳 What is the reimbursement policy?",
                on_click=handle_query,
                args=("What is the reimbursement policy?",),
                use_container_width=True,
            )
        with col2:
            st.button(
                "✈️ How do I request time off?",
                on_click=handle_query,
                args=("How do I request time off?",),
                use_container_width=True,
            )
            st.button(
                "🏥 What are the employee benefits?",
                on_click=handle_query,
                args=("What are the employee benefits?",),
                use_container_width=True,
            )
else:
    welcome_placeholder.empty()


# ==========================================
# Render Chat Messages
# ==========================================
if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown('<div class="msg-header-user">USER</div>', unsafe_allow_html=True)
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="📖"):
                st.markdown(
                    """
                    <div class="msg-header-ai">
                        <span class="msg-role-title">AI HANDBOOK</span>
                        <span class="rag-tag">RAG-GROUNDED RESPONSE</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(message["content"])
                if message.get("sources"):
                    render_sources_card(message["sources"])


# ==========================================
# Process Pending User Query via Real RAG Pipeline
# ==========================================
if "pending_prompt" in st.session_state:
    pending_prompt = st.session_state.pop("pending_prompt")

    with st.chat_message("assistant", avatar="📖"):
        st.markdown(
            """
            <div class="msg-header-ai">
                <span class="msg-role-title">AI HANDBOOK</span>
                <span class="rag-tag">RAG-GROUNDED RESPONSE</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Initial ChatGPT-style 3-dot loading animation
        answer_placeholder = st.empty()
        answer_placeholder.markdown(
            """
            <div class="typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        try:
            # ------------------------------------------
            # Execute Real RAG Pipeline (Retrieval + Stream)
            # ------------------------------------------
            response_data = ask_question_stream(pending_prompt)

            stream = response_data.get("stream")
            sources = response_data.get("sources", [])

            # Stream LLM tokens live into the Streamlit UI (replaces 3 dots)
            if stream:
                answer = answer_placeholder.write_stream(stream)
            else:
                answer = "I couldn't find that information in the company policies."
                answer_placeholder.markdown(answer)

            # Render Sources if information was found
            is_not_found = "couldn't find that information in the company policies" in answer.lower()
            valid_sources = [] if is_not_found else sources

            if valid_sources:
                render_sources_card(valid_sources)

            # Save to session history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": valid_sources,
            })

        except Exception as e:
            error_message = "⚠️ Something went wrong while processing your question."
            st.error(error_message)
            st.code(str(e))