
import argparse
import sys

import spacy
from spacy.tokens import Doc

MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-ca-ner"

_hf_pipeline = None
_nlp_blank = None


def get_hf_pipeline():
    global _hf_pipeline

    if _hf_pipeline is None:
        from transformers import pipeline

        print(
            f"Loading {MODEL_NAME} (first call only, then cached)...",
            file=sys.stderr
        )

        _hf_pipeline = pipeline(
            "token-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            aggregation_strategy="simple",
            stride=128
        )

    return _hf_pipeline

def get_blank_nlp():
    global _nlp_blank
    if _nlp_blank is None:
        _nlp_blank = spacy.blank("ar")
        _nlp_blank.add_pipe("sentencizer")
    return _nlp_blank


def annotate(text: str) -> Doc:
    """Run the transformer NER model on `text` and return a spaCy Doc with .ents set."""
    nlp = get_blank_nlp()
    doc = nlp(text)
   
    hf_ner = get_hf_pipeline()
    results = hf_ner(text)

    spans = []
    for r in results:
        span = doc.char_span(
            r["start"],
            r["end"],
            label=r["entity_group"],
            alignment_mode="expand"
        )
        if span is not None:
            spans.append(span)

    # spaCy rejects overlapping spans; keep the longest span when two overlap.
    spans = spacy.util.filter_spans(spans)
    doc.ents = spans
    return doc
    

def main():
    parser = argparse.ArgumentParser(description="Model B: transformer Arabic NER")
    parser.add_argument("text", nargs="?", default=None)
    args = parser.parse_args()

    text = args.text or (
        "توجه وزير الخارجية إلى دمشق لعقد اجتماع مع الأمم المتحدة"
    )

    doc = annotate(text)
    print(f"{'ENTITY':<40} LABEL")
    print("-" * 50)
    for ent in doc.ents:
        print(f"{ent.text:<40} {ent.label_}")


if __name__ == "__main__":
    main()
