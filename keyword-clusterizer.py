import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Keyword Clusterizer", layout="wide")
st.title("Keyword Clustering Tool")

# === STEP 1: Upload File
uploaded_file = st.file_uploader("Upload your CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.subheader("Preview of your file:")
    st.dataframe(df.head())

    # STEP 2: Determine clustering mode first (outside the form)
    st.subheader("Clustering Settings")
    clustering_mode = st.radio("Clustering mode", ["Auto (Silhouette Score)", "Manual"])

    # Inside the form
    with st.form("clustering_form"):
        text_cols = st.multiselect("Select column(s) to use for clustering", df.columns.tolist())

        manual_k = None
        if clustering_mode == "Manual":
            manual_k = st.number_input("Enter number of clusters", min_value=2, step=1, value=10)

        submitted = st.form_submit_button("Run Clustering")

    # === STEP 3: If form submitted, run clustering
    if submitted:
        if not text_cols:
            st.error("Please select at least one column to cluster on.")
        else:
            # Combine columns
            df_text = df[text_cols].astype(str).fillna("")
            df_text["combined"] = df_text.apply(lambda row: " ".join(row), axis=1)
            texts = df_text["combined"].tolist()

            # Generate embeddings
            st.info("Generating semantic embeddings with MiniLM...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(texts, show_progress_bar=True)

            # Auto or manual cluster count
            if clustering_mode == "Auto (Silhouette Score)":
                st.info("Finding best number of clusters using Silhouette Score...")
                scores = []
                for k in range(2, 21):  # You can change the auto test range here
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = km.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)
                    scores.append((k, score))
                n_clusters = max(scores, key=lambda x: x[1])[0]
                st.success(f"Best number of clusters found: {n_clusters}")
            else:
                n_clusters = manual_k
                st.info(f"Using manually entered number of clusters: {n_clusters}")

            # KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(embeddings)

            # Assign human-readable cluster labels based on centroid-nearest example
            centroids = kmeans.cluster_centers_
            similarities = cosine_similarity(embeddings, centroids)
            cluster_labels = []

            for i in range(n_clusters):
                cluster_indices = np.where(df['cluster'] == i)[0]
                cluster_sims = similarities[cluster_indices, i]
                best_index = cluster_indices[np.argmax(cluster_sims)]
                cluster_labels.append(df_text.iloc[best_index]["combined"])

            label_map = {i: label for i, label in enumerate(cluster_labels)}
            df['cluster_label'] = df['cluster'].map(label_map)

            # === Show results
            st.subheader("Clustered Results")
            st.dataframe(df[[*text_cols, 'cluster_label']].head(20))

            # === Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Clustered CSV", csv, "semantic_clusters.csv", "text/csv")
