
import pandas as pd
import numpy as np
import networkx as nx
import joblib


# ---------------------------------
# LOAD TRAINED ML MODEL
# ---------------------------------

def load_model():
    bundle = joblib.load("models/fraud_model.joblib")

    return bundle["model"], bundle["feature_columns"]


# ---------------------------------
# PREPARE TRANSACTION FEATURES
# ---------------------------------

def prepare_transaction(data, feature_columns):

    amount = float(data["amount"])

    features = {
        "step": int(data["step"]),
        "amount_log": np.log1p(amount),
        "oldbalanceOrg": float(data["oldbalanceOrg"]),
        "newbalanceOrig": float(data["newbalanceOrig"]),
        "oldbalanceDest": float(data["oldbalanceDest"]),
        "newbalanceDest": float(data["newbalanceDest"]),

        "origin_balance_error": (
            float(data["oldbalanceOrg"])
            - amount
            - float(data["newbalanceOrig"])
        ),

        "destination_balance_error": (
            float(data["newbalanceDest"])
            - float(data["oldbalanceDest"])
            - amount
        )
    }

    transaction_type = data["type"]

    for col in feature_columns:
        if col.startswith("type_"):
            features[col] = int(
                col == f"type_{transaction_type}"
            )

    df = pd.DataFrame([features])

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return df


# ---------------------------------
# GRAPH ANALYSIS
# ---------------------------------

def analyze_transaction(data, original_graph):

    G = original_graph.copy()

    sender = str(data["nameOrig"])
    receiver = str(data["nameDest"])
    amount = float(data["amount"])

    # Add current transaction to graph
    G.add_edge(
        sender,
        receiver,
        amount=amount,
        type=data.get("type", "TRANSFER")
    )

    # Calculate graph features
    graph_features = {
        "degree": int(G.degree(sender)),
        "in_degree": int(G.in_degree(sender)),
        "out_degree": int(G.out_degree(sender)),
        "receiver_degree": int(G.degree(receiver)),
        "receiver_in_degree": int(G.in_degree(receiver)),
        "receiver_out_degree": int(G.out_degree(receiver))
    }

    # Find connected accounts around sender and receiver
    nodes = {sender, receiver}
    frontier = {sender, receiver}

    for _ in range(2):

        next_frontier = set()

        for node in frontier:

            if node in G:

                neighbors = (
                    set(G.predecessors(node))
                    | set(G.successors(node))
                )

                next_frontier.update(neighbors)

        new_nodes = next_frontier - nodes

        nodes.update(new_nodes)
        frontier = new_nodes

        if len(nodes) >= 50:
            break

    # Always preserve sender and receiver
    priority_nodes = [sender, receiver]

    # Select the most connected neighboring accounts
    other_nodes = sorted(
        nodes - {sender, receiver},
        key=lambda node: G.degree(node),
        reverse=True
    )

    selected_nodes = set(
        priority_nodes + other_nodes[:48]
    )

    subgraph = G.subgraph(selected_nodes).copy()

    return {
        "graph_features": graph_features,
        "graph": subgraph
    }


# ---------------------------------
# COMPLETE TRANSACTION PREDICTION
# ---------------------------------

def predict_transaction(data, original_graph):

    model, feature_columns = load_model()

    # Prepare ML input
    X = prepare_transaction(
        data,
        feature_columns
    )

    # Predict fraud
    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    fraud_probability = probabilities[
        list(model.classes_).index(1)
    ]

    # Perform graph analysis
    graph_result = analyze_transaction(
        data,
        original_graph
    )

    return {
        "prediction": int(prediction),
        "fraud_probability": float(fraud_probability),
        "graph_features": graph_result["graph_features"],
        "graph": graph_result["graph"]
    }
