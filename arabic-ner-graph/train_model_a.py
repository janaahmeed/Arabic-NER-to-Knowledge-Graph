import argparse
import subprocess
import sys
from pathlib import Path
import spacy
from spacy.tokens import Doc, DocBin
from datasets import load_dataset
from spacy.training.iob_utils import iob_to_biluo, biluo_tags_to_spans

nlp = spacy.blank("ar")
ROOT = Path(__file__).resolve().parent
TRAIN_DATA = ROOT / "data" / "train.spacy"
DEV_DATA = ROOT / "data" / "dev.spacy"
OUTPUT_DIR = ROOT / "model-a" 
CONFIG_PATH= ROOT / "data" / "config.cfg"


def reconstruct_sentences(words, tags, max_tokens=40):
    """Groups flat word-by-word lists into sentences based on punctuation or max length."""
    sentences, sent_tags = [], []
    curr_words, curr_tags = [], []

    for w, t in zip(words, tags):
        curr_words.append(w)
        curr_tags.append(t)

        # End sentence at terminal punctuation or max token limit
        if w in [".", "!", "؟"] or len(curr_words) >= max_tokens:
            sentences.append(curr_words)
            sent_tags.append(curr_tags)
            curr_words, curr_tags = [], []

    if curr_words:
        sentences.append(curr_words)
        sent_tags.append(curr_tags)

    return sentences, sent_tags

def repair_bio_tags(tags: list[str]) -> list[str]:
    repaired, prev_type = [], None

    for tag in tags:
        if tag.startswith("I-"):
            curr_type = tag.split("-", 1)[1]
            # If previous type doesn't match current type, convert I- to B-
            tag = tag if prev_type == curr_type else f"B-{curr_type}"
            prev_type = curr_type
        else:
            prev_type = tag.split("-", 1)[1] if tag.startswith("B-") else None

        repaired.append(tag)

    return repaired


def make_docbin(split, nlp) -> DocBin:
    db = DocBin()
    sentences, sent_tags = reconstruct_sentences(split["word"], split["tag"])
    for tokens, tags in zip(sentences, sent_tags):
        if not tokens:
            continue
        doc = Doc(nlp.vocab, words=tokens)
        re_tags = repair_bio_tags(tags)
        biluo = iob_to_biluo(re_tags) # IOB2 > BILUO (spaCy's internal scheme)
        spans =biluo_tags_to_spans(doc, biluo) # token boundries
        doc.ents = spans
        db.add(doc)
    return db


def prepare_data():
   if TRAIN_DATA.exists() and DEV_DATA.exists():
        print("Using existing cached data...")
        return
   else:
    print("Loading asas-ai/ANERCorp from Hugging Face (cached after first run)...")
    ds = load_dataset("asas-ai/ANERCorp")
    ds.save_to_disk(ROOT/ "data" / "ANERCorp")
    #print(f"Saved in {SAVE_PATH}")
    print(type(ds["train"].features["tag"]))
    all_tags_labels = [tag.split( "_", 1)[1] if "_" in tag else tag  for sentence in ds["train"]["tag"] for tag in sentence]
    raw_names = list(set(all_tags_labels))

    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Converting train split...")
    train_db = make_docbin(ds["train"], nlp)
    train_db.to_disk(TRAIN_DATA)    
    # dedicated validation split exists.
    
    test_split_name = "validation" if "validation" in ds else "test"
    print(f"Converting {test_split_name} split...")
    dev_db = make_docbin(ds[test_split_name], nlp)
    dev_db.to_disk(DEV_DATA)

    print(f"Saved {TRAIN_DATA} and {DEV_DATA}")


def ensure_config():
    if CONFIG_PATH.exists():
        return
    print("Generating base training config...")
    subprocess.run(
    [
        sys.executable, "-m", "spacy", "init", "config",
        str(CONFIG_PATH),
        "--lang", "ar",
        "--pipeline", "ner",
        "--optimize", "efficiency",   
        "--force",
    ],
    check=True,
)


def train():
    ensure_config()
    print("Training Model A (this can take a while on CPU)...")
    subprocess.run(
        [
            sys.executable, "-m", "spacy", "train",
            str(CONFIG_PATH),
            "--paths.train", str(TRAIN_DATA),
            "--paths.dev", str(DEV_DATA),
            "--output", str(OUTPUT_DIR),
        ],
        check=True,
    )
    print(f"Done. Load the model with: spacy.load('{OUTPUT_DIR / 'model-best'}')")

def main():
    parser = argparse.ArgumentParser(description="Train Model A: spaCy NER on ANERCorp")
    parser.add_argument("--prep-only", action="store_true", help="only cache data/*.spacy")
    parser.add_argument("--train-only", action="store_true", help="skip data prep")
    args = parser.parse_args()

    if not args.train_only:
        prepare_data()
    if not args.prep_only:
        train()


if __name__ == "__main__":
    main()
