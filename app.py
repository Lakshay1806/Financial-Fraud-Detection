
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

from utils.transaction_analysis import predict_transaction


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FinGuard AI | Financial Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = "dataset/PS_20174392719_1491204439457_log.csv"


# =====================================================
# PREMIUM DARK THEME
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #0b1220 !important;
    color: #e8eef8 !important;
}

header[data-testid="stHeader"] {
    background: #0b1220 !important;
}

.main .block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 2.5rem;
}

section[data-testid="stSidebar"] {
    background: #101b2d !important;
    border-right: 1px solid #26364d;
}

section[data-testid="stSidebar"] * {
    color: #dce6f5 !important;
}

h1, h2, h3, h4, h5 {
    color: #f4f7fc !important;
}

p, label, .stMarkdown {
    color: #c3cee0;
}

.hero {
    background: linear-gradient(120deg, #17263d, #142238);
    border: 1px solid #344b6b;
    border-radius: 20px;
    padding: 30px 34px;
    margin-bottom: 26px;
    box-shadow: 0 14px 35px rgba(0,0,0,.20);
}

.hero h1 {
    color: #f8fbff !important;
    font-size: 2.15rem;
    margin-bottom: 8px;
}

.hero p {
    color: #c5d4e8 !important;
}

.section-card {
    background: #142238;
    border: 1px solid #2c405c;
    border-radius: 16px;
    padding: 20px 22px;
    margin: 12px 0;
}

div[data-testid="stMetric"] {
    background: #16263c;
    border: 1px solid #304763;
    border-radius: 14px;
    padding: 16px 18px;
}

div[data-testid="stMetric"] label {
    color: #aebfd5 !important;
}

div[data-testid="stMetricValue"] {
    color: #f3f7ff !important;
}

.stButton > button {
    background: #2563eb !important;
    color: white !important;
    border: 1px solid #4b8cf0 !important;
    border-radius: 10px !important;
    min-height: 46px;
    font-weight: 650;
}

.stButton > button:hover {
    background: #1d4ed8 !important;
}

div[data-baseweb="select"] > div,
.stTextInput input {
    background: #1b2b42 !important;
    color: #f4f7fc !important;
    border: 1px solid #405775 !important;
    border-radius: 9px !important;
}

div[data-baseweb="popover"],
div[role="listbox"] {
    background: #1b2b42 !important;
}

div[data-baseweb="menu"] * {
    color: #f4f7fc !important;
}

.stAlert {
    background: #17283f !important;
    border: 1px solid #344b68 !important;
    color: #e7eef9 !important;
}

hr {
    border-color: #30435d !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #30445f;
    border-radius: 10px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, nrows=100000)


@st.cache_resource
def build_base_graph():

    graph_df = pd.read_csv(
        DATA_PATH,
        nrows=100000,
        usecols=[
            "type",
            "nameOrig",
            "nameDest",
            "amount"
        ]
    )

    graph_df = graph_df[
        graph_df["type"].isin(["TRANSFER", "CASH_OUT"])
    ].copy()

    return nx.from_pandas_edgelist(
        graph_df,
        source="nameOrig",
        target="nameDest",
        edge_attr=["amount", "type"],
        create_using=nx.DiGraph()
    )


# =====================================================
# CHART THEME
# =====================================================

def make_dark_chart(fig, height=420):

    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#142238",
        plot_bgcolor="#142238",
        font=dict(color="#e5edf8"),
        margin=dict(l=25, r=25, t=55, b=25)
    )

    return fig


# =====================================================
# GRAPH VISUALIZATION
# =====================================================

def create_graph_visualization(
    graph,
    sender,
    receiver,
    max_nodes=50
):

    graph = graph.copy()

    graph.add_node(sender)
    graph.add_node(receiver)

    seeds = {sender, receiver}
    nearby = set(seeds)
    frontier = set(seeds)

    for _ in range(2):

        next_frontier = set()

        for node in frontier:

            if node in graph:

                next_frontier.update(
                    graph.predecessors(node)
                )

                next_frontier.update(
                    graph.successors(node)
                )

        next_frontier -= nearby

        nearby.update(next_frontier)
        frontier = next_frontier

        if len(nearby) >= max_nodes:
            break

    extras = sorted(
        nearby - seeds,
        key=lambda node: graph.degree(node),
        reverse=True
    )

    chosen = list(seeds) + extras[:max_nodes - len(seeds)]

    subgraph = graph.subgraph(chosen).copy()

    # Show the currently selected sender-to-receiver relationship.
    if not subgraph.has_edge(sender, receiver):
        subgraph.add_edge(sender, receiver)

    pos = nx.spring_layout(
        subgraph,
        seed=17,
        k=0.85,
        iterations=45
    )

    edge_x = []
    edge_y = []

    for source, target in subgraph.edges():

        x0, y0 = pos[source]
        x1, y1 = pos[target]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.2, color="#526985"),
        hoverinfo="none"
    )

    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    hover = []
    labels = []

    for node in subgraph.nodes():

        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)
        labels.append(str(node))

        if node == sender:
            node_colors.append("#f59e0b")
            node_sizes.append(23)

        elif node == receiver:
            node_colors.append("#a78bfa")
            node_sizes.append(23)

        else:
            node_colors.append("#38bdf8")
            node_sizes.append(14)

        hover.append(
            f"Account: {node}<br>"
            f"Incoming: {subgraph.in_degree(node)}<br>"
            f"Outgoing: {subgraph.out_degree(node)}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=9, color="#dce8f7"),
        hovertext=hover,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=1.3, color="#e7eef8")
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace]
    )

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        visible=False
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        visible=False
    )

    make_dark_chart(fig, 520)

    fig.update_layout(
        showlegend=False,
        title="Transaction Network Graph"
    )

    return fig, subgraph


# =====================================================
# DROPDOWN HELPERS
# =====================================================

def unique_values(data, column):

    values = data[column].dropna().unique().tolist()

    try:
        values = sorted(values)
    except TypeError:
        values = sorted(values, key=str)

    return values


def number_format(value):

    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("# 🛡️ FinGuard AI")
    st.caption("FINANCIAL INTELLIGENCE PLATFORM")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Overview",
            "📊 Transaction Analysis",
            "📈 Dataset Dashboard"
        ]
    )

    st.divider()

    st.markdown("### Project Information")
    st.caption("AI-Based Financial Fraud Detection")
    st.caption("Random Forest • NetworkX")
    st.caption("PaySim Transaction Dataset")

    st.divider()

    st.caption("Machine Learning | Graph Analytics")


# =====================================================
# LOAD DATA
# =====================================================

try:
    df = load_data()

except Exception as exc:

    st.error(
        f"Dataset could not be loaded from {DATA_PATH}: {exc}"
    )

    st.stop()


# =====================================================
# OVERVIEW
# =====================================================

if page == "🏠 Overview":

    st.markdown("""
    <div class="hero">
        <h1>🛡️ FinGuard AI</h1>
        <p>Intelligent Financial Fraud Detection System</p>
        <p>
        Identify suspicious financial activity with machine learning,
        transaction patterns and account-network analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.header("Financial Intelligence Dashboard")

    total = len(df)
    fraud_count = int(df["isFraud"].sum())
    legitimate_count = total - fraud_count

    c1, c2, c3 = st.columns(3)

    c1.metric("Transactions in Sample", f"{total:,}")
    c2.metric("Fraud Transactions", f"{fraud_count:,}")
    c3.metric("Legitimate Transactions", f"{legitimate_count:,}")

    st.divider()

    st.subheader("Explore the Platform")

    left, right = st.columns(2)

    with left:

        st.markdown("""
        <div class="section-card">
            <h3>📊 Transaction Analysis</h3>
            <p>
            Independently select transaction variables from the dataset,
            estimate fraud risk and explore connected accounts.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="section-card">
            <h3>📈 Dataset Dashboard</h3>
            <p>
            Review transaction types, amount patterns and
            fraud distribution in the dataset sample.
            </p>
        </div>
        """, unsafe_allow_html=True)

    type_counts = df["type"].value_counts()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                hole=0.58
            )
        ]
    )

    st.subheader("Transaction Type Distribution")

    st.plotly_chart(
        make_dark_chart(fig, 400),
        use_container_width=True
    )


# =====================================================
# TRANSACTION ANALYSIS
# =====================================================

elif page == "📊 Transaction Analysis":

    st.markdown("""
    <div class="hero">
        <h1>📊 Transaction Analysis</h1>
        <p>
        Independently select transaction variables and analyze
        financial risk using machine learning and graph analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Every dropdown contains values taken from the PaySim dataset. "
        "You can select each variable independently. "
        "The selected combination does not have to be an existing complete row."
    )

    required_columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:

        st.error(
            f"Required dataset columns are missing: {missing_columns}"
        )

        st.stop()

    st.subheader("1. Transaction Information")

    col1, col2, col3 = st.columns(3)

    type_options = unique_values(df, "type")
    step_options = unique_values(df, "step")
    amount_options = unique_values(df, "amount")

    with col1:

        selected_type = st.selectbox(
            "Transaction Type",
            options=type_options,
            format_func=str,
            key="selected_type"
        )

    with col2:

        selected_step = st.selectbox(
            "Transaction Step",
            options=step_options,
            format_func=lambda x: f"{int(x):,}",
            key="selected_step"
        )

    with col3:

        selected_amount = st.selectbox(
            "Transaction Amount",
            options=amount_options,
            format_func=number_format,
            key="selected_amount"
        )

    st.divider()

    st.subheader("2. Account Information")

    col1, col2 = st.columns(2)

    sender_options = unique_values(df, "nameOrig")
    receiver_options = unique_values(df, "nameDest")

    with col1:

        selected_sender = st.selectbox(
            "Sender Account ID",
            options=sender_options,
            format_func=str,
            key="selected_sender"
        )

    with col2:

        selected_receiver = st.selectbox(
            "Receiver Account ID",
            options=receiver_options,
            format_func=str,
            key="selected_receiver"
        )

    st.caption(
        "Account IDs are searchable. Start typing an ID in the dropdown."
    )

    st.divider()

    st.subheader("3. Sender Balance Details")

    col1, col2 = st.columns(2)

    old_org_options = unique_values(df, "oldbalanceOrg")
    new_org_options = unique_values(df, "newbalanceOrig")

    with col1:

        selected_old_org = st.selectbox(
            "Sender Previous Balance",
            options=old_org_options,
            format_func=number_format,
            key="selected_old_org"
        )

    with col2:

        selected_new_org = st.selectbox(
            "Sender New Balance",
            options=new_org_options,
            format_func=number_format,
            key="selected_new_org"
        )

    st.divider()

    st.subheader("4. Receiver Balance Details")

    col1, col2 = st.columns(2)

    old_dest_options = unique_values(df, "oldbalanceDest")
    new_dest_options = unique_values(df, "newbalanceDest")

    with col1:

        selected_old_dest = st.selectbox(
            "Receiver Previous Balance",
            options=old_dest_options,
            format_func=number_format,
            key="selected_old_dest"
        )

    with col2:

        selected_new_dest = st.selectbox(
            "Receiver New Balance",
            options=new_dest_options,
            format_func=number_format,
            key="selected_new_dest"
        )

    st.divider()

    st.subheader("Selected Transaction Variables")

    selected_summary = pd.DataFrame({
        "Variable": [
            "Transaction Type",
            "Step",
            "Amount",
            "Sender ID",
            "Receiver ID",
            "Sender Old Balance",
            "Sender New Balance",
            "Receiver Old Balance",
            "Receiver New Balance"
        ],
        "Selected Value": [
            str(selected_type),
            str(selected_step),
            number_format(selected_amount),
            str(selected_sender),
            str(selected_receiver),
            number_format(selected_old_org),
            number_format(selected_new_org),
            number_format(selected_old_dest),
            number_format(selected_new_dest)
        ]
    })

    st.dataframe(
        selected_summary,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "All selected values originate from their respective dataset columns. "
        "The combined selection may not represent one historical transaction."
    )

    analyze = st.button(
        "Analyze Transaction",
        use_container_width=True
    )

    if analyze:

        transaction = {
            "step": int(selected_step),
            "type": str(selected_type),
            "amount": float(selected_amount),
            "nameOrig": str(selected_sender),
            "nameDest": str(selected_receiver),
            "oldbalanceOrg": float(selected_old_org),
            "newbalanceOrig": float(selected_new_org),
            "oldbalanceDest": float(selected_old_dest),
            "newbalanceDest": float(selected_new_dest)
        }

        try:

            with st.spinner(
                "Analyzing selected variables and account connections..."
            ):

                base_graph = build_base_graph()

                result = predict_transaction(
                    transaction,
                    base_graph
                )

            prediction = int(result["prediction"])
            probability = float(result["fraud_probability"])
            features = result["graph_features"]

            st.divider()

            st.header("Analysis Results")

            c1, c2 = st.columns([1.5, 1])

            if prediction == 1:
                c1.error("⚠️ HIGH FRAUD RISK DETECTED")
            else:
                c1.success("✓ LOW FRAUD RISK")

            c2.metric(
                "Estimated Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(
                max(0.0, min(1.0, probability))
            )

            st.caption(
                "This is the model's estimated score, "
                "not proof that a transaction is fraudulent."
            )

            st.divider()

            st.header("Graph Analysis")

            st.caption(
                "Orange: Sender | Purple: Receiver | Blue: Connected accounts"
            )

            a, b, c = st.columns(3)

            a.metric(
                "Sender Degree",
                features["degree"]
            )

            b.metric(
                "Sender In-Degree",
                features["in_degree"]
            )

            c.metric(
                "Sender Out-Degree",
                features["out_degree"]
            )

            a, b, c = st.columns(3)

            a.metric(
                "Receiver Degree",
                features["receiver_degree"]
            )

            b.metric(
                "Receiver In-Degree",
                features["receiver_in_degree"]
            )

            c.metric(
                "Receiver Out-Degree",
                features["receiver_out_degree"]
            )

            fig, subgraph = create_graph_visualization(
                result["graph"],
                transaction["nameOrig"],
                transaction["nameDest"]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            a, b = st.columns(2)

            a.metric(
                "Accounts Displayed",
                subgraph.number_of_nodes()
            )

            b.metric(
                "Connections Displayed",
                subgraph.number_of_edges()
            )

            st.divider()

            st.subheader("Transaction Summary")

            summary = pd.DataFrame({
                "Parameter": [
                    "Sender",
                    "Receiver",
                    "Transaction Type",
                    "Amount",
                    "Step",
                    "Prediction",
                    "Estimated Fraud Probability"
                ],
                "Value": [
                    transaction["nameOrig"],
                    transaction["nameDest"],
                    transaction["type"],
                    f"{transaction['amount']:,.2f}",
                    transaction["step"],
                    "Fraud risk" if prediction == 1 else "Low fraud risk",
                    f"{probability * 100:.2f}%"
                ]
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

        except Exception as exc:

            st.error(
                f"Transaction analysis failed: {exc}"
            )

            st.info(
                "Check that models/fraud_model.joblib exists "
                "and utils/transaction_analysis.py returns "
                "the expected result fields."
            )


# =====================================================
# DATASET DASHBOARD
# =====================================================

elif page == "📈 Dataset Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>📈 Dataset Dashboard</h1>
        <p>
        Explore transaction statistics, financial activity
        and fraud distribution.
        </p>
    </div>
    """, unsafe_allow_html=True)

    total = len(df)
    fraud_count = int(df["isFraud"].sum())
    legitimate_count = total - fraud_count

    c1, c2, c3 = st.columns(3)

    c1.metric("Transactions in Sample", f"{total:,}")
    c2.metric("Fraud Transactions", f"{fraud_count:,}")
    c3.metric("Legitimate Transactions", f"{legitimate_count:,}")

    st.divider()

    st.subheader("Fraud Distribution")

    fraud_fig = go.Figure(
        data=[
            go.Pie(
                labels=["Fraud", "Legitimate"],
                values=[fraud_count, legitimate_count],
                hole=0.58
            )
        ]
    )

    st.plotly_chart(
        make_dark_chart(fraud_fig, 420),
        use_container_width=True
    )

    st.divider()

    st.subheader("Transaction Type Analysis")

    type_data = (
        df.groupby("type")
        .size()
        .reset_index(name="Count")
    )

    type_fig = go.Figure(
        data=[
            go.Bar(
                x=type_data["type"],
                y=type_data["Count"],
                text=type_data["Count"],
                textposition="auto"
            )
        ]
    )

    type_fig.update_layout(
        xaxis_title="Transaction Type",
        yaxis_title="Number of Transactions"
    )

    st.plotly_chart(
        make_dark_chart(type_fig, 420),
        use_container_width=True
    )

    st.divider()

    st.subheader("Transaction Amount Distribution")

    amount_fig = go.Figure(
        data=[
            go.Histogram(
                x=df["amount"],
                nbinsx=50
            )
        ]
    )

    amount_fig.update_layout(
        xaxis_title="Transaction Amount",
        yaxis_title="Frequency"
    )

    st.plotly_chart(
        make_dark_chart(amount_fig, 420),
        use_container_width=True
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.drop(
            columns=["isFraud", "isFlaggedFraud"],
            errors="ignore"
        ),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;padding:14px;color:#8294ad;">
        FinGuard AI | Financial Fraud Detection System
        <br>
        Machine Learning • Random Forest • Graph Analytics
    </div>
    """,
    unsafe_allow_html=True
)
