from spacy.tokens import DocBin
import argparse
import sys
import spacy
from spacy.pipeline import EntityRuler  

"""spaCy ships no pretrained Arabic NER model, so this builds a blank Arabic
pipeline and adds hand-written patterns keyed on common Arabic news triggers
(titles for people, prefixes for organizations, prepositions for locations).
"""

PERSON_TITLES = ["الرئيس", "الدكتور", "الوزير", "السيد", "المهندس", "الأستاذ", "الشيخ"]
ORG_PREFIXES = ["شركة", "مؤسسة", "وزارة", "جامعة", "منظمة", "هيئة", "بنك"]
LOC_PREFIXES = ["مدينة", "محافظة", "قرية", "إقليم"]
custom_patterns = [
    
    # Single-token skills (exact string match)
    {
        "label": "SKILL",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "python",
                        "sql",
                        "c++",
                        "c#",
                        "java",
                        "pytorch",
                        "tensorflow",
                    ]
                }
            }
        ],
    },
    # Multi-token skills (defined token by token)
    {"label": "SKILL", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "data"}, {"LOWER": "science"}]},
    {"label": "SKILL", "pattern": [{"LOWER": "agentic"}, {"LOWER": "ai"}]},
   #------------------------------------------------------------
    #  English titles (e.g., "Senior Software Engineer", "CEO", "Junior Developer")
    {
        "label": "JOB_TITLE",
        "pattern": [
            {
                "LOWER": {
                    "IN": [
                        "senior",
                        "junior",
                        "juniour",
                        "lead",
                        "principal",
                        "ceo",
                        "cto",
                        "cfo",
                    ]
                }
            },
            {"IS_ALPHA": True, "OP": "{0,2}"},  # Allows optional words like "Software"
        ],
    },
    # Arabic titles + domain (e.g., "مهندس برمجيات", "مطور الذكاء الاصطناعي")
    {
        "label": "JOB_TITLE",
        "pattern": [
            {"TEXT": {"IN": ["مهندس", "مدير", "مطور", "قائد", "رئيس"]}},
            {"IS_PUNCT": False, "OP": "{1,2}"},  # Matches 1-2 words following title
        ],
    },
]

def build_patterns():
    ARABIC_PREPOSITIONS = ARABIC_PREPOSITIONS = ["في", "من", "إلى", "على", "عن", "مع", "بسبب", "حتى"]
    patterns = []
    for title in PERSON_TITLES:
      patterns.append(
            {
                "label": "PER",
                "pattern": [
                    {"TEXT": title},
                    {
                        "IS_ALPHA": True,
                        "IS_STOP": False,
                        "TEXT": {"NOT_IN": ARABIC_PREPOSITIONS},
                        "OP": "{1,3}",
                    },
                ],
            }
        )
    for prefix in ORG_PREFIXES:
       patterns.append(
            {
                "label": "ORG",
                "pattern": [
                    {"TEXT": prefix},
                    {
                        "IS_ALPHA": True,
                        "IS_STOP": False,
                        "TEXT": {"NOT_IN": ARABIC_PREPOSITIONS},
                        "OP": "{1,3}",
                    },
                ],
            }
        )
    for prefix in LOC_PREFIXES:
        patterns.append(
            {
                "label": "LOC",
                "pattern": [
                    {"TEXT": prefix},
                    {
                        "IS_ALPHA": True,
                        "IS_STOP": False,
                        "TEXT": {"NOT_IN": ARABIC_PREPOSITIONS},
                        "OP": "{1,2}",
                    },
                ],
            }
        )
    patterns.extend(custom_patterns)
    return patterns


def build_nlp():
    """Build and return a blank Arabic pipeline with the EntityRuler installed."""
    nlp = spacy.blank("ar")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(build_patterns())
    return nlp

DEMO_TEXT = (
    "زار الرئيس عبد الفتاح السيسي جامعة القاهرة يوم الاثنين، "
    "والتقى بوزير التعليم في مقر وزارة التربية والتعليم بمدينة القاهرة."
)


def main():
    parser = argparse.ArgumentParser(description="Rule-based Arabic NER ")
    parser.add_argument("text", nargs="?", default=None, help="Arabic text to analyze")
    parser.add_argument("--file", type=str, default=None, help="Path to a UTF-8 text file")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = DEMO_TEXT
        print("(no input given — running on the built-in demo sentence)\n", file=sys.stderr)

    nlp = build_nlp()
    raw_text = " ".join(args.text.split()) if args.text else DEMO_TEXT
    doc = nlp(raw_text)

    if not doc.ents:
        print("No entities matched. Try extending PERSON_TITLES / ORG_PREFIXES / LOC_PREFIXES.")
        return

    print(f"{'ENTITY':<40} LABEL")
    print("-" * 50)
    for ent in doc.ents:
        print(f"{ent.text:<40} {ent.label_}")


if __name__ == "__main__":
    main()
