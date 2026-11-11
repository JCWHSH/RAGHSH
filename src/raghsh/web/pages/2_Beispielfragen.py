"""
Beispielfragen — anklickbare Einstiegsfragen für den HsH-Assistenten.

Ein Klick legt die Frage in ``st.session_state.pending_frage`` ab und springt
zurück zur Chat-Seite, die sie dort aufgreift und direkt stellt.
"""

import streamlit as st

st.set_page_config(
    page_title="Beispielfragen — HsH-Assistent",
    page_icon="💡",
    layout="wide",
)

# Pfad relativ zum Streamlit-Einstiegsskript
CHAT_SEITE = "app.py"

BEISPIELFRAGEN: list[tuple[str, list[str]]] = [
    ("👤 Personen & Räume", [
        "Wo ist das Büro von Martin Grotjahn?",
        "Wer ist Ansprechpartner für Prüfungsfragen?",
    ]),
    ("🎓 Studium & Praxisphase", [
        "Kann ich eine Praxisphase an der Fakultät 1 in Teilzeit machen?",
        "Darf ich in der Vorlesung Eis essen?",
    ]),
    ("🛠️ Services vor Ort", [
        "Wo kann ich in der Hochschule Poster drucken?",
    ]),
    ("🤖 KI an der HsH", [
        "Wann und wo findet das KI Forum 2026 statt?",
        "Welche Initiativen gibt es um KI Kompetenz zu verstärken?",
    ]),
]

st.markdown(
    """
<style>
div[data-testid="stVerticalBlock"] div.stButton > button {
    width: 100%;
    text-align: left;
    justify-content: flex-start;
    white-space: normal;
    height: auto;
    padding: 0.75rem 1rem;
    border: 1px solid #c8dff3;
    background: #f4f8fd;
    color: #1a1a1a;
    border-radius: 8px;
    font-weight: 500;
}
div[data-testid="stVerticalBlock"] div.stButton > button:hover {
    border-color: #00559a;
    background: #e8f0fb;
    color: #00559a;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("💡 Beispielfragen")
st.caption(
    "Klicken Sie eine Frage an — sie wird in das Fragenfeld des Chats "
    "übernommen und dort direkt beantwortet."
)

for gruppe, fragen in BEISPIELFRAGEN:
    st.subheader(gruppe)
    for frage in fragen:
        if st.button(frage, key=f"bsp_{frage}", use_container_width=True):
            st.session_state.pending_frage = frage
            st.switch_page(CHAT_SEITE)

st.divider()
st.caption(
    "Die Fragen decken unterschiedliche Quellenlagen ab: von der "
    "Personensuche über Ordnungen und Serviceseiten bis zu Fragen, "
    "auf die es in der Wissensdatenbank bewusst keine Antwort gibt."
)
