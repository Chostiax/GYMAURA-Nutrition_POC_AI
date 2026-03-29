# GymAura Nutrition PoC

## Overview
This Proof of Concept (PoC) validates whether natural language meal descriptions can be converted into structured food data using an internal USDA-based dataset.

The goal is to test the feasibility of a conversational logging pipeline before integrating mobile input or AI-based approaches.

## Scope
This PoC focuses on the internal text pipeline only:
1. Text input (natural language sentences)
2. Food extraction (rule-based)
3. Dataset matching (USDA-based dataset)
4. Nutrition calculation (calories, protein, carbs, fat)

## Pipeline
For each input sentence, the system performs:
1. Extraction — Detects food items and simple quantities from natural language
2. Matching — Maps foods to dataset entries using alias normalization, exact match, token overlap, and fuzzy fallback
3. Nutrition Calculation — Estimates portion size (grams) and scales per-100g values
4. Aggregation — Computes total calories and macros

## KPIs
Supervisor-defined targets:
- Food detection accuracy ≥ 80%
- Dataset matching success ≥ 85%
- Quantity detection accuracy (simple cases) ≥ 60%
- End-to-end success ≥ 70%

## Results
Evaluation performed on 29 natural-language test sentences:

- Food detection accuracy: 92.9%
- Dataset matching success: 92.9%
- Quantity detection accuracy: 82.8%
- End-to-end success: 86.2%

All KPI targets are met.

## How to Run
Requirements:
- Python 3.10+

Install dependencies:
python -m pip install -r requirements.txt

Run the project:
python main.py

This will:
- Load the dataset
- Run a demo example
- Evaluate all test cases
- Print metrics and detailed results

## Project Structure

```bash
GYMAURA-NUTRITION-POC/
│
├── data/
│   └── food_item_rows.csv
│
├── src/
│   ├── data_prep.py
│   ├── extractor.py
│   ├── matcher.py
│   ├── nutrition.py
│   ├── pipeline.py
│   └── evaluation.py
│
├── tests/
│   └── test_cases.py
│
├── main.py
├── README.md
└── requirements.txt
```

## Design Decisions
- Python-first approach for rapid iteration
- Rule-based extraction to build a stable baseline
- Strict evaluation to ensure realistic performance
- Semi-curated rules to improve robustness without overfitting

## Limitations
- Quantity estimation handles only simple cases
- Nutrition is computed only when portion size can be estimated
- Generic food matching can still be imperfect
- Extraction is rule-based and not fully robust to all conversational inputs

## Next Steps
- Improve matcher ranking for generic foods
- Expand portion estimation coverage
- Compare with AI-based approach (accuracy vs cost)
- Integrate with mobile (Flutter + speech-to-text)
