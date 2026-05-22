import streamlit as st
import requests

# ── Config page ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ticket Support — Priorisation & Routage",
    page_icon="🎫",
    layout="centered"
)

API_URL = "http://127.0.0.1:8000/predict"

# ── CSS custom ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #01696f;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #0c4e54; }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-top: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
    }
    .priority-critical { background-color: #f8d7da; color: #842029; border: 1px solid #f5c2c7; }
    .priority-high     { background-color: #fff3cd; color: #664d03; border: 1px solid #ffecb5; }
    .priority-medium   { background-color: #cff4fc; color: #055160; border: 1px solid #b6effb; }
    .priority-low      { background-color: #d1e7dd; color: #0f5132; border: 1px solid #badbcc; }
    .routing-box       { background-color: #e8f4f8; color: #01696f;  border: 1px solid #b8dce8; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🎫 Ticket Support — Priorisation & Routage")
st.markdown("Saisissez les informations du ticket pour obtenir une **prédiction de priorité** et de **catégorie de routage**.")
st.divider()

# ── Formulaire ─────────────────────────────────────────────────────────────
with st.form("ticket_form"):
    st.subheader("📝 Informations du ticket")

    text = st.text_area(
        "Description du ticket *",
        placeholder="Ex: My laptop screen is broken and I need urgent assistance...",
        height=120
    )

    col1, col2 = st.columns(2)

    with col1:
        customer_gender = st.selectbox(
            "Genre du client",
            ["Male", "Female", "Other"]
        )
        product_purchased = st.selectbox(
            "Produit concerné",
            ["Laptop", "Smartphone", "Tablet", "Smartwatch",
             "TV", "Headphones", "Camera", "Software", "Other"]
        )
        customer_age = st.number_input(
            "Âge du client",
            min_value=18, max_value=100, value=35
        )

    with col2:
        ticket_type = st.selectbox(
            "Type de ticket",
            ["Technical support", "Billing inquiry",
             "Product inquiry", "Refund request",
             "Cancellation request", "Other"]
        )
        ticket_channel = st.selectbox(
            "Canal de contact",
            ["Email", "Phone", "Chat", "Social media"]
        )
        model_name = st.selectbox(
            "Modèle à utiliser",
            ["custom", "pretrained"],
            help="custom = BiLSTM | pretrained = TF-IDF"
        )

    submitted = st.form_submit_button("🔍 Analyser le ticket")

# ── Prédiction ─────────────────────────────────────────────────────────────
if submitted:
    if not text.strip():
        st.warning("⚠️ Veuillez saisir une description du ticket.")
    else:
        payload = {
            "text": text,
            "Customer_Gender": customer_gender,
            "Product_Purchased": product_purchased,
            "Ticket_Type": ticket_type,
            "Ticket_Channel": ticket_channel,
            "Customer_Age": float(customer_age),
            "model_name": model_name
        }

        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()

                st.divider()
                st.subheader("📊 Résultats de l'analyse")

                col_p, col_r = st.columns(2)

                priority = result.get("predicted_priority", "N/A").lower()
                routing  = result.get("predicted_routing", "N/A")

                # Couleur selon priorité
                priority_class = {
                    "critical": "priority-critical",
                    "high":     "priority-high",
                    "medium":   "priority-medium",
                    "low":      "priority-low"
                }.get(priority, "priority-medium")

                priority_emoji = {
                    "critical": "🔴",
                    "high":     "🟠",
                    "medium":   "🟡",
                    "low":      "🟢"
                }.get(priority, "⚪")

                with col_p:
                    st.markdown("**🚨 Priorité prédite**")
                    st.markdown(
                        f'''<div class="result-box {priority_class}">
                            {priority_emoji} {result["predicted_priority"]}
                        </div>''',
                        unsafe_allow_html=True
                    )

                with col_r:
                    st.markdown("**📂 Catégorie de routage**")
                    st.markdown(
                        f'''<div class="result-box routing-box">
                            📁 {routing.replace("_", " ").title()}
                        </div>''',
                        unsafe_allow_html=True
                    )

                st.markdown(f"<p style='color:gray;font-size:0.85rem;margin-top:1rem;'>Modèle utilisé : <b>{result['model_used']}</b></p>", unsafe_allow_html=True)

                # Détails dans un expander
                with st.expander("🔎 Voir les détails de la requête"):
                    st.json(payload)

            except requests.exceptions.ConnectionError:
                st.error("❌ Impossible de se connecter à l'API. Vérifiez que le backend FastAPI tourne sur `http://127.0.0.1:8000`.")
            except requests.exceptions.Timeout:
                st.error("⏱️ Le modèle met trop de temps à répondre. Réessayez.")
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:gray;font-size:0.8rem;'>Projet IA — Routage intelligent de tickets support | Backend : FastAPI | Modèles : TensorFlow/Keras</p>",
    unsafe_allow_html=True
)
