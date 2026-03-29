from pprint import pprint

from src.data_prep import load_and_prepare_dataset
from src.pipeline import run_pipeline

DATA_PATH = "data/food_item_rows.csv"


def main():
    print("Loading dataset...")
    dataset = load_and_prepare_dataset(DATA_PATH)
    print(f"Loaded {len(dataset)} usable dataset rows.")

    text = input("\nEnter a meal description: ").strip()
    if not text:
        print("No input provided.")
        return

    result = run_pipeline(text, dataset, model="gpt-5-mini")

    print("\n" + "=" * 60)
    print("AI PIPELINE RESULT")
    print("=" * 60)
    pprint(result)


if __name__ == "__main__":
    main()