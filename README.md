# GymAura Nutrition PoC (AI Version)

## Goal
This PoC extends the baseline nutrition pipeline by integrating an AI model for natural language parsing.

The objective is to evaluate whether using an LLM improves food extraction from real user input while keeping the rest of the pipeline unchanged.

## Scope

This version replaces the rule-based extractor with an AI-based parser.

Pipeline:

text input  
→ AI parsing (LLM → structured JSON)  
→ dataset matching (USDA-based)  
→ nutrition calculation  

## Key Difference from Baseline

Baseline:
- rule-based extraction (regex / heuristics)

AI version:
- LLM parses sentence into structured JSON
- same matcher and nutrition logic are reused

This allows a fair comparison between both approaches.

## AI Parsing

The AI model:
- extracts food items
- extracts quantities and units
- interprets "a/an" as quantity = 1
- maps vague quantities like "some" to default values (e.g. 100g)
- can optionally decompose composite dishes (e.g. "caesar salad") into main ingredients

Example:

Input:
"I ate a big caesar salad and 2 tomatoes"

Output:
- lettuce (200g)
- chicken (150g)
- croutons (50g)
- parmesan (30g)
- caesar dressing (70g)
- tomato (2)

## Dataset Matching

Each extracted (or inferred) food is matched against the internal USDA dataset using:
- normalization
- exact matching
- token overlap
- fuzzy matching fallback

## Nutrition Calculation

Nutrition is computed from the dataset:
- calories
- protein
- carbs
- fat

Even when the AI infers ingredients, the nutritional values come from the dataset.

## Models Used

- gpt-5-mini (default, more accurate)
- gpt-5-nano (cheaper alternative)

The model can be selected in the UI.

## Token Usage

Each request consumes tokens:
- depends on sentence length and model
- used to estimate cost per request


## Running the Project

Install dependencies:

pip install -r requirements.txt

Run Streamlit UI:

python -m streamlit run app.py

## Summary

This AI PoC demonstrates that:
- AI can significantly improve food extraction from natural input
- the existing dataset + nutrition pipeline can remain unchanged
- the main trade-off is between flexibility and cost/control