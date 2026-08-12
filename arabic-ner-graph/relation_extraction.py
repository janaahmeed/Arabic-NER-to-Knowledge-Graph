"""
relation_extraction.py
------------------------
Turns a spaCy Doc's tagged entities (from any of the three NER backends in
this project) into relation triples: (entity1, type1, relation, entity2, type2).

"""

import argparse
import re
from itertools import combinations
import spacy
from ner_rule_based import build_nlp


TRIGGERS = {
    "PER_ORG": {
        "relation": "affiliated_with",
        "phrases": ["رئيس", "وزير", "مدير", "عضو في", "يعمل في", "التحق ب", "رئيسة"],
    },
    "PER_PER": {
        "relation": "interacts_with",
        "phrases": ["التقى", "اجتمع مع", "زار", "تحدث إلى", "التقت"],
    },
    "ORG_MISC": {  # dates often land in MISC when tagged by ANERCorp-trained models
        "relation": "dated_event",
        "phrases": ["تأسست في", "أُعلن في", "افتتحت في", "أقيم في", "في عام"],
    },
    "ORG_LOC": {
        "relation": "located_in",
        "phrases": ["في", "الكائن في", "ومقرها"],
    },
}

# Sentence-terminator fallback for pipelines without a sentencizer/parser.
_SENT_SPLIT_RE = re.compile(r"[.!؟\n]+")


def get_sentence_char_spans(doc):
    """Return a list of (start_char, end_char) sentence spans for `doc`.

    Uses doc.sents when the pipeline set sentence boundaries; otherwise falls
    back to a regex split on Arabic/Latin sentence terminators.
    """
    if doc.has_annotation("SENT_START"):
        return [(s.start_char, s.end_char) for s in doc.sents]

    spans = []
    pos = 0
    for match in _SENT_SPLIT_RE.finditer(doc.text):
        end = match.start()
        if end > pos:
            spans.append((pos, end))
        pos = match.end()
    if pos < len(doc.text):
        spans.append((pos, len(doc.text)))
    return spans


def _entities_in_span(doc, start_char, end_char):
    return [e for e in doc.ents if e.start_char >= start_char and e.end_char <= end_char]


def cooccurrence_edges(doc):
    """Weak baseline: any two entities sharing a sentence 'co-occur'."""
    edges = []
    for start, end in get_sentence_char_spans(doc):
        ents = _entities_in_span(doc, start, end)
        for e1, e2 in combinations(ents, 2):
            edges.append((e1.text, e1.label_, "co-occurs-with", e2.text, e2.label_))
    return edges


def extract_typed_relations(doc):
    """Trigger-word relations: scan the text between two co-occurring entities."""
    relations = []
    for start, end in get_sentence_char_spans(doc):
        ents = _entities_in_span(doc, start, end)
        for e1, e2 in combinations(ents, 2):
            first, second = (e1, e2) if e1.start_char <= e2.start_char else (e2, e1)
            between = doc.text[first.end_char:second.start_char]

            pair_key = f"{first.label_}_{second.label_}"
            rule = TRIGGERS.get(pair_key) or TRIGGERS.get(f"{second.label_}_{first.label_}")
            if rule and any(p in between for p in rule["phrases"]):
                relations.append(
                    (first.text, first.label_, rule["relation"], second.text, second.label_)
                )
    return relations


def extract_all(doc, include_cooccurrence=True):
    """Convenience: typed relations, optionally padded with co-occurrence edges
    for entity pairs that didn't match a trigger phrase."""
    typed = extract_typed_relations(doc)
    if not include_cooccurrence:
        return typed

    typed_pairs = {(t[0], t[3]) for t in typed} | {(t[3], t[0]) for t in typed}
    extra = [
        edge for edge in cooccurrence_edges(doc)
        if (edge[0], edge[3]) not in typed_pairs
    ]
    return typed + extra


DEMO_TEXT = (
    "أعلن رئيس شركة النيل للاتصالات أن الشركة تأسست في عام 1998 بمدينة القاهرة. "
    "والتقى الرئيس التنفيذي بوزير الاتصالات لبحث خطط التوسع."
)


def main():
    parser = argparse.ArgumentParser(description="Extract entity relations from Arabic text")
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = DEMO_TEXT

    # Use the rule-based pipeline here for a self-contained demo; swap in
    # train_model_a's saved model or model_b_transformer.annotate() as needed.
    
    nlp = build_nlp()
    doc = nlp(text)

    relations = extract_all(doc)
    if not relations:
        print("No relations found. Check that entities were tagged and triggers match.")
        return

    print(f"{'ENTITY 1':<25}{'TYPE':<6}{'RELATION':<18}{'ENTITY 2':<25}TYPE")
    print("-" * 90)
    for e1, t1, rel, e2, t2 in relations:
        print(f"{e1:<25}{t1:<6}{rel:<18}{e2:<25}{t2}")


if __name__ == "__main__":
    main()
