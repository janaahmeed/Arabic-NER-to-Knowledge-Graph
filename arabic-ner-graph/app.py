from pathlib import Path

import spacy
import streamlit as st
import streamlit.components.v1 as components
from spacy import displacy
from pyvis.network import Network
from ner_rule_based import build_nlp as build_rule_based_nlp
from relation_extraction import extract_all
from build_graph import build_graph, COLOR_MAP
# Model B: transformer
from model_b_transformer import annotate


MODEL_A_PATH = Path(__file__).resolve().parent / "model-a" / "model-best"


ENT_COLORS = {"PER": "#F0997B", "PERS": "#F0997B", "ORG": "#AFA9EC", "LOC": "#5DCAA5", "MISC": "#B4B2A9"}

st.set_page_config(page_title="Arabic NER to  Graph", layout="wide")
st.title("Arabic NER to  Graph")


text = st.text_area("try  your Arabic  text",  height=180)

model_choice = st.radio(
    "NER backend",
    ["Rule-based (EntityRuler)", "Model A: CNN trained on ANERCorp", "Model B: transformer (CAMeLBERT-NER)"],
    horizontal=True,
)


@st.cache_resource
def get_rule_based_nlp():
    return build_rule_based_nlp()


@st.cache_resource
def get_model_a_nlp():
    if not MODEL_A_PATH.exists():
        return None
    return spacy.load('D:\\arabic-ner-graph\\arabic-ner-graph\\model-a\\model-best')


def run_ner(text: str, choice: str):
    if choice.startswith("Rule-based"):
        nlp = get_rule_based_nlp()
        return nlp(text)
    if choice.startswith("Model A"):
        nlp = get_model_a_nlp()
        if nlp is None:
            st.error(
                f"No trained model found at {MODEL_A_PATH}. "
                "Run `python train_model_a.py` first."
            )
            st.stop()
        return nlp(text)
    
    elif choice.startswith("Model B"):
           if choice.startswith("Model B"):
               return annotate(text)
    
           return annotate(text)


if st.button("Analyze", type="primary") and text.strip():
    with st.spinner("Running NER..."):
        doc = run_ner(text, model_choice)

    st.subheader("Tagged entities")
    if doc.ents:
        html = displacy.render(doc, style="ent", options={"colors": ENT_COLORS}, page=False)
        components.html(f'<div dir="rtl" style="font-size:20px; line-height:2;">{html}</div>', height=220, scrolling=True)
    else:
        st.info("No entities detected.")

    relations = extract_all(doc) if doc.ents else []

    st.subheader("Extracted relations")
    if relations:
        st.table(
            [
                {"Entity 1": e1, "Type 1": t1, "Relation": rel, "Entity 2": e2, "Type 2": t2}
                for e1, t1, rel, e2, t2 in relations
            ]
        )
    else:
        st.info("No relations extracted (need at least two entities in the same sentence).")

    if relations:
        st.subheader("Knowledge graph")
        G = build_graph(relations)
        

        net = Network(directed=True, cdn_resources="in_line", height="700px", width="100%")
        for node, attrs in G.nodes(data=True):
            net.add_node(node, label=node, color=COLOR_MAP.get(attrs.get("type"), "#888888"))
        for u, v, attrs in G.edges(data=True):
            net.add_edge(u, v, label=attrs.get("label", ""))

        graph_path = "graph.html"
        original_open = open

# Generate the PyVis HTML
        with open(graph_path, "w", encoding="utf-8") as f:
           f.write(net.generate_html())

# Display it in Streamlit
        with open(graph_path, "r", encoding="utf-8") as f:
              html = f.read()

        
        components.html(open(graph_path, encoding="utf-8").read(), height=520)
