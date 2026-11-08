
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from utils.transaction_analysis import analyze_transaction, predict_transaction


# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="AI Financial Fraud Detection",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 AI-Based Financial Fraud Detection")
st.markdown("### Intelligent Transaction Analysis & Graph Monitoring System")


# ---------------------------------
# LOAD DATASET
# ---------------------------------

DATA_PATH = "dataset/PS_20174392719_1491204439457_log.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, nrows=100000)


@st.cache_resource
def build_base_graph():

    df = load_data()
    G = nx.DiGraph()

    transactions = df[
        df["type"].isin(["TRANSFER", "CASH_OUT"])
    ]

    for row in transactions.itertuples(index=False):

        G.add_edge(
            row.nameOrig,
            row.nameDest,
            amount=float(row.amount)
        )

    return G


@st.cache_data
def get_statistics():

    df = load_data()

    total = len(df)
    fraud = int(df["isFraud"].sum())

    return total, fraud, total - fraud


# ---------------------------------
# GRAPH VISUALIZATION
# ---------------------------------

def create_graph_visualization(G):

    if G.number_of_nodes() == 0:
        return None

    nodes = list(G.nodes())[:50]
    H = G.subgraph(nodes).copy()

    if H.number_of_nodes() == 0:
        return None

    pos = nx.spring_layout(H, seed=42)

    edge_x = []
    edge_y = []

    for source, destination in H.edges():

        x0, y0 = pos[source]
        x1, y1 = pos[destination]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#64748b"),
        hoverinfo="none"
    )

    node_x = [pos[node][0] for node in H.nodes()]
    node_y = [pos[node][1] for node in H.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(node) for node in H.nodes()],
        textposition="top center",
        marker=dict(
            size=15,
            color=[
                "#ef4444" if H.degree(node) > 2 else "#38bdf8"
                for node in H.nodes()
            ],
            line=dict(width=1, color="white")
        ),
        hovertext=[
            f"Account: {node}<br>Degree: {H.degree(node)}"
            for node in H.nodes()
        ],
        hoverinfo="text"
    )

    fig = go.Figure(
        data=[edge_trace, node_trace]
    )

    fig.update_layout(
        title="Transaction Network Graph",
        showlegend=False,
        height=550,
        template="plotly_dark",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    return fig


# ---------------------------------
# SIDEBAR
# ---------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    ["Transaction Analysis", "Dataset Dashboard"]
)


# ---------------------------------
# DATASET DASHBOARD
# ---------------------------------

if page == "Dataset Dashboard":

    total, fraud, legitimate = get_statistics()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Fraud Transactions", f"{fraud:,}")
    col3.metric("Legitimate Transactions", f"{legitimate:,}")

    df = load_data()

    st.subheader("Fraud Distribution")

    chart = pd.DataFrame({
        "Type": ["Legitimate", "Fraud"],
        "Count": [legitimate, fraud]
    })

    fig = px.pie(
        chart,
        names="Type",
        values="Count",
        hole=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Transaction Amount Analysis")

    fig = px.histogram(
        df.sample(min(10000, len(df)), random_state=42),
        x="amount",
        color="isFraud"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------
# TRANSACTION ANALYSIS
# ---------------------------------

elif page == "Transaction Analysis":

    st.subheader("Enter Transaction Details")

    st.info(
        "Enter transaction information to analyze its "
        "financial risk and account network."
    )

    with st.form("transaction_form"):

        col1, col2 = st.columns(2)

        with col1:

            sender = st.text_input(
                "Sender Account ID",
                value="C100001"
            )

            amount = st.number_input(
                "Transaction Amount",
                min_value=0.01,
                value=50000.0
            )

            old_origin = st.number_input(
                "Sender Previous Balance",
                min_value=0.0,
                value=100000.0
            )

            new_origin = st.number_input(
                "Sender New Balance",
                min_value=0.0,
                value=50000.0
            )

        with col2:

            receiver = st.text_input(
                "Receiver Account ID",
                value="C100002"
            )

            transaction_type = st.selectbox(
                "Transaction Type",
                ["TRANSFER", "CASH_OUT"]
            )

            old_destination = st.number_input(
                "Receiver Previous Balance",
                min_value=0.0,
                value=20000.0
            )

            new_destination = st.number_input(
                "Receiver New Balance",
                min_value=0.0,
                value=70000.0
            )

        step = st.number_input(
            "Transaction Time Step",
            min_value=1,
            value=100
        )

        submitted = st.form_submit_button(
            "Analyze Transaction",
            use_container_width=True
        )

    if submitted:

        if not sender.strip() or not receiver.strip():

            st.error("Please enter both account IDs.")

        elif sender.strip() == receiver.strip():

            st.error("Sender and receiver must be different accounts.")

        else:

            transaction = {
                "step": int(step),
                "type": transaction_type,
                "amount": float(amount),
                "nameOrig": sender.strip(),
                "nameDest": receiver.strip(),
                "oldbalanceOrg": float(old_origin),
                "newbalanceOrig": float(new_origin),
                "oldbalanceDest": float(old_destination),
                "newbalanceDest": float(new_destination)
            }

            try:

                with st.spinner(
                    "Analyzing transaction and account network..."
                ):

                    G = build_base_graph()

                    result = predict_transaction(
                        transaction,
                        G
                    )

                prediction = result["prediction"]
                probability = result["fraud_probability"]
                graph_features = result["graph_features"]
                transaction_graph = result["graph"]

                st.divider()

                st.header("Analysis Results")

                col1, col2 = st.columns(2)

                with col1:

                    if prediction == 1:

                        st.error(
                            "⚠️ FRAUDULENT TRANSACTION DETECTED"
                        )

                    else:

                        st.success(
                            "✅ LEGITIMATE TRANSACTION"
                        )

                with col2:

                    st.metric(
                        "Fraud Probability",
                        f"{probability * 100:.2f}%"
                    )

                st.progress(
                    max(0.0, min(1.0, float(probability)))
                )

                # GRAPH FEATURES

                st.divider()

                st.header("Graph Analysis")

                st.markdown(
                    "Account connectivity and transaction network features"
                )

                feature_cols = st.columns(3)

                feature_cols[0].metric(
                    "Degree",
                    graph_features["degree"]
                )

                feature_cols[1].metric(
                    "In-Degree",
                    graph_features["in_degree"]
                )

                feature_cols[2].metric(
                    "Out-Degree",
                    graph_features["out_degree"]
                )

                # TRANSACTION GRAPH

                st.subheader("Account Connection Visualization")

                st.caption(
                    "Red nodes indicate accounts with higher connectivity "
                    "in this displayed network, not confirmed fraud."
                )

                fig = create_graph_visualization(
                    transaction_graph
                )

                if fig:

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:

                    st.warning("No graph connections available.")

                # TRANSACTION DETAILS

                st.divider()

                st.subheader("Transaction Information")

                st.json({
                    "Sender": sender,
                    "Receiver": receiver,
                    "Amount": amount,
                    "Transaction Type": transaction_type,
                    "Prediction": (
                        "Fraud" if prediction == 1 else "Legitimate"
                    ),
                    "Fraud Probability": round(
                        float(probability) * 100, 2
                    )
                })

            except Exception as e:

                st.error(f"Analysis Error: {e}")

                st.warning(
                    "Check model file, transaction analysis code "
                    "and input features."
                )
