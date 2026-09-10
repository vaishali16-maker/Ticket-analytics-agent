import streamlit as st
import requests
import plotly.express as px
import pandas as pd

API_URL = "https://ticket-analytics-agent.onrender.com"
st.set_page_config(
    page_title="Ticket SLA Analytics Agent",
    page_icon="🎫",
    layout="wide"
)


st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { color: #1a1a2e; }
    .stButton button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton button:hover {
        background-color: #4338ca;
    }
</style>
""", unsafe_allow_html=True)
st.title("🎫 Ticket SLA Analytics Agent")
st.caption("Ask questions in plain English, predict SLA breach risk, and explore trends")


# Session state for conversation memory
if "history" not in st.session_state:
    st.session_state.history = []
tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "⚠️ Predict New Ticket", "📊 Dashboard"])


# Tab 1: NL-to-SQL chat with conversation memory
with tab1:
    st.subheader("Ask a question about your tickets")
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What is the SLA breach rate by priority?",
        label_visibility="collapsed"
    )

    col_ask, col_clear = st.columns([1, 5])
    with col_ask:
        ask_clicked = st.button("Ask", type="primary")
    with col_clear:
        if st.button("Clear conversation"):
            st.session_state.history = []
            st.rerun()
    if ask_clicked and question:
        with st.spinner("Thinking..."):
            response = requests.post(f"{API_URL}/ask", json={
                "question": question,
                "history": st.session_state.history
            })
            data = response.json()

        if "error" in data and "result" not in data:
            st.error(data["error"])
        else:
            st.success(f"✓ Answered in {data.get('attempts_needed', 1)} attempt(s)")
            with st.expander("View generated SQL"):
                st.code(data["final_sql"], language="sql")
            st.dataframe(data["result"], use_container_width=True)
            st.session_state.history.append({
                "question": question,
                "answer_summary": str(data.get("result", ""))[:200]
            })
    if st.session_state.history:
        st.divider()
        st.markdown("**Conversation history**")
        for h in st.session_state.history[-5:]:
            st.markdown(f"- **Q:** {h['question']}")


# Tab 2: SLA breach prediction
with tab2:
    st.subheader("Predict SLA breach risk for a new ticket")
    col1, col2 = st.columns(2)
    with col1:
        priority = st.selectbox("Priority", ["low", "medium", "high", "urgent"])
        channel = st.selectbox("Channel", ["chat", "email", "in_app", "phone_transcript", "web_form"])
        customer_segment = st.selectbox("Customer Segment",
            ["individual", "small_business", "enterprise", "education", "non_profit"])
        sla_plan = st.selectbox("SLA Plan", ["standard", "gold", "platinum"])

    with col2:
        product_area = st.selectbox("Product Area",
            ["billing", "login_auth", "api_integration", "analytics_dashboard",
             "mobile_app", "notifications", "data_export"])
        issue_type = st.text_input("Issue Type", value="bug")
        region = st.selectbox("Region", ["EU", "NA", "APAC", "LATAM", "MEA"])

    if st.button("Predict Risk", type="primary"):
        payload = {
            "priority": priority, "channel": channel,
            "customer_segment": customer_segment, "product_area": product_area,
            "issue_type": issue_type, "region": region, "sla_plan": sla_plan
        }
        with st.spinner("Predicting..."):
            response = requests.post(f"{API_URL}/predict", json=payload)
            data = response.json()
        risk_pct = data["breach_probability"] * 100
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Breach Risk", f"{risk_pct:.1f}%")
        with col_b:
            if data["predicted_breach"]:
                st.error("⚠️ This ticket is at risk of breaching SLA")
            else:
                st.success("✅ Likely to meet SLA")
        fig = px.bar(
            x=[risk_pct], y=[""], orientation="h",
            range_x=[0, 100],
            color_discrete_sequence=["#ef4444" if risk_pct > 50 else "#10b981"]
        )
        fig.update_layout(
            height=100, showlegend=False,
            xaxis_title="Risk %", yaxis_visible=False,
            margin=dict(l=0, r=0, t=10, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)


# Tab 3: Dashboard
with tab3:
    st.subheader("Ticket Analytics Dashboard")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Tickets", "100,000")
    k2.metric("Overall Breach Rate", "34.6%")
    k3.metric("Best Model Recall", "81%", help="Balanced Logistic Regression")
    k4.metric("Strongest Signal", "Priority", help="16%→51% breach rate range")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**SLA Breach Rate by Priority**")
        priority_df = pd.DataFrame({
            "priority": ["urgent", "high", "medium", "low"],
            "breach_rate": [0.160, 0.164, 0.292, 0.509]
        })
        fig1 = px.bar(
            priority_df, x="priority", y="breach_rate",
            color="breach_rate", color_continuous_scale="Reds",
            text_auto=".1%"
        )
        fig1.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("**Model Comparison (Breached class)**")
        model_df = pd.DataFrame({
            "model": ["LR (unbalanced)", "LR (balanced)", "Random Forest"] * 2,
            "metric": ["Recall"] * 3 + ["Precision"] * 3,
            "value": [0.40, 0.81, 0.52, 0.72, 0.51, 0.54]
        })
        fig2 = px.bar(
            model_df, x="model", y="value", color="metric",
            barmode="group",
            color_discrete_map={"Recall": "#4f46e5", "Precision": "#a5b4fc"}
        )
        st.plotly_chart(fig2, use_container_width=True)