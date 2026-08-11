
## Project Title

**Arabic NER to Knowledge Graph: From Entity Recognition to Relationship Modeling**
# Arabic NER → Relations → Knowledge Graph

## Overview

An Arabic NLP pipeline that transforms Arabic news text into structured entities, relationships, and an interactive knowledge graph.

The project compares three NER approaches:

1. **Rule-Based NER** using spaCy `EntityRuler`
2. **Model A** — a spaCy NER model trained from scratch on **ANERCorp**
3. **Model B** — the Arabic **CAMeLBERT-NER** transformer

Detected entities are passed to a lightweight relation-extraction layer and then converted into a **NetworkX MultiDiGraph**, visualized interactively with **pyvis**. A **Streamlit** application provides an end-to-end interface for comparing the NER backends.

## Architecture

```text
Arabic Article
      ↓
NER Backend
 ┌───────────────┬────────────────┬────────────────────┐
 │ Rule-Based    │ spaCy Model A  │ CAMeLBERT Model B  │
 │ EntityRuler   │ ANERCorp       │ Transformer NER    │
 └───────────────┴────────────────┴────────────────────┘
      ↓
Entities + Labels
      ↓
Relation Extraction
 ├── Sentence co-occurrence
 └── Trigger-word based typed relations
      ↓
Relation Triples
      ↓
NetworkX MultiDiGraph
      ↓
pyvis Interactive Graph
      ↓
HTML / GraphML / Streamlit UI
```

## Technologies & Libraries

- **Python** — main implementation language
- **spaCy** — NLP pipeline, `EntityRuler`, NER training, `Doc` representation
- **Hugging Face Datasets** — ANERCorp loading and caching
- **Hugging Face Transformers / CAMeLBERT** — transformer-based Arabic NER
- **NetworkX** — knowledge graph construction
- **pyvis** — interactive graph visualization
- **Streamlit** — web interface
- **GraphML** — graph interchange/export format

## NER Approaches

### 1. Rule-Based NER

Uses spaCy `EntityRuler` with customizable Arabic patterns and lists such as:

- `PERSON_TITLES`
- `ORG_PREFIXES`
- `LOC_PREFIXES`

**Advantages:** transparent, fast, easy to extend.

**Challenges:** rules can be sensitive to ambiguity, spelling variation, and unseen linguistic patterns.

### 2. Model A — spaCy + ANERCorp

A statistical NER pipeline trained from scratch on **ANERCorp**.

main points in trianing :

1. Download/cache ANERCorp.
2. Convert IOB-tagged tokens into spaCy `Doc` objects.
3. Convert entity tags to spaCy spans.
4. Save `data/train.spacy` and `data/dev.spacy`.
5. Generate `config.cfg`.
6. Train the NER pipeline.
7. Save the best checkpoint to `model-a/model-best`.

The supplied training log reached approximately:

- **ENTS_F:** 59.30%
- **ENTS_P:** 75.25%
- **ENTS_R:** 59.00%

These values are the recorded development-set metrics from the supplied training run.

### 3. Model B — CAMeLBERT-NER

Uses:

`CAMeL-Lab/bert-base-arabic-camelbert-ca-ner`

The transformer is loaded from **Hugging Face Hub ** and wrapped into a spaCy `Doc` so that it can participate in the same application workflow.

## Data & Labeling

ANERCorp uses:

- `PER` — Person
- `LOC` — Location
- `ORG` — Organization
- `MISC` — Miscellaneous

The project uses IOB/BIO-style sequence labels during data preparation. spaCy internally works with the richer **BILUO** representation.

NER depends on surface forms, prefixes, suffixes, and contextual information.

## Arabic NLP Challenges Addressed

### Ambiguity

A surface form can represent different entity types depending on context.

### Orthographic Variation

Arabic text may contain different spellings or missing diacritics/letters.

### Entity Boundaries

Multi-token entities must end at the correct boundary.
tested include stopping an entity before:

- punctuation
- unrelated following words
- numbers

### Adjacent Entities

The system must distinguish multiple entities appearing close together in the same sentence.

## Relation Extraction

The project uses a lightweight, dependency-parser-free strategy.

### Layer 1 — Sentence Co-occurrence

### Layer 2 — Trigger-Based Typed Relations

The result is a set of relation triples that can be represented as graph edges.

## Knowledge Graph

The graph is built with **NetworkX** using a `MultiDiGraph`.

Both relationships can coexist.

### Visualization

The graph is rendered as an interactive HTML visualization with **pyvis**.

### Export

The graph can be exported to **GraphML**, making it usable with graph tools such as:

- Neo4j
- Gephi
- yEd
- igraph
- Cytoscape

## Testing Strategy

The project includes tests for:

- Entity ambiguity
- Arabic orthographic variation
- Entity boundaries
- Technical/custom entities
- Adjacent entities

The custom examples include concepts such as:

- `JOB_TITLE`
- `SKILL`

## Evaluation Metrics

The spaCy training output includes:

- **ENTS_F** — entity F1 score
- **ENTS_P** — entity precision
- **ENTS_R** — entity recall
- **LOSS TOK2VEC** — Tok2Vec training loss
- **LOSS NER** — NER training loss
- **SCORE** — checkpoint selection score

## Streamlit Application

Run:

```bash
streamlit run app.py
```

## Project Structure

```text
arabic-ner-graph/
├── data/
├── ner_rule_based.py
├── train_model_a.py
├── model_b_transformer.py
├── relation_extraction.py
├── build_graph.py
├── app.py
├── requirements.txt
└── README.md
```

## Quick Start

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Rule-based NER:

```bash
python ner_rule_based.py "زار الرئيس عبد الفتاح السيسي جامعة القاهرة يوم الاثنين"
```

Train Model A:

```bash
python train_model_a.py
```

Prepare data only:

```bash
python train_model_a.py --prep-only
```

Train using existing prepared data:

```bash
python train_model_a.py --train-only
```

Run Model B:

```bash
python model_b_transformer.py "توجه وزير الخارجية إلى دمشق لعقد اجتماع مع الأمم المتحدة"
```

Extract relations:

```bash
python relation_extraction.py
```

Build the graph:

```bash
python build_graph.py --out graph.html
```

Export GraphML:

```bash
python build_graph.py --export graph.graphml
```

## Engineering Value

This project demonstrates an end-to-end **Arabic NLP and information extraction workflow**:

It combines deterministic rules, supervised NLP, transformer-based Arabic NER, graph modeling, testing, and application development into one practical pipeline.
