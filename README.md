# GPT-4o LOVES ROCK

An empirical analysis of GPT-4o's Rock-Paper-Scissors selection bias.

## Executive Summary

Our experiments conclusively demonstrate that **GPT-4o exhibits a strong and persistent bias toward selecting "Rock"** when playing Rock-Paper-Scissors, regardless of how the question is phrased or the order in which options are presented.

- **Overall Selection**: 82.67% Rock, 11.33% Scissors, 6.00% Paper
- This bias persists across different prompt formulations and option orderings
- The bias is statistically significant with a sample size of 150 total trials

## Key Findings

### Finding 1: Overwhelming "Rock" Preference

GPT-4o selected "Rock" in 82.67% of all trials, with "Scissors" at 11.33% and "Paper" at just 6.00%. This preference held regardless of prompt phrasing or option ordering.

### Finding 2: Option Ordering Had Minimal Impact

| Prompt Variant | Rock % | Scissors % | Paper % | Rock Probability |
|----------------|--------|------------|---------|------------------|
| rock, paper, scissors | 72.00% | 22.00% | 6.00% | 59.34% |
| paper, rock, scissors | 88.00% | 10.00% | 2.00% | 67.17% |
| scissors, rock, paper | 88.00% | 2.00% | 10.00% | 51.62% |

Surprisingly, putting "Rock" first actually resulted in the most balanced distribution, though still heavily Rock-biased.

### Finding 3: Token-Level Probability Analysis

At the token level:
- "Rock" tokens consistently had the highest probabilities (avg ~0.59)
- "Paper" tokens had the lowest probabilities (avg ~0.05)
- "Scissors" tokens had slightly higher probabilities than "Paper" (avg ~0.09)

### Finding 4: Bias Persists Regardless of Phrasing

We tested various phrasings, from direct questions to conversational prompts, and found the bias remained consistent.

## Visualizations

The experiment generated several visualization files:
- `rps_analysis_order_fixed.png` - Breakdown by prompt variant
- `rps_analysis_order_overall_fixed.png` - Aggregate results across all prompts

## Implications

This experiment reveals a non-trivial bias in GPT-4o's behavior for what should be a random choice. Possible explanations include:

1. **Training Data Bias**: Rock might appear first or more frequently in training examples of Rock-Paper-Scissors
2. **Tokenization Effects**: "Rock" may tokenize more efficiently than other options
3. **Cultural Patterns**: The phrase "Rock, Paper, Scissors" (with Rock first) is the standard ordering in English

## Technical Details

- **Model**: GPT-4o (2024-08-06)
- **Top-p**: 1.0
- **Sample Size**: 150 trials (50 per prompt variant)
- **Seed**: Not fixed (to ensure randomness)

## Code Reference

The full experiment code is available in `rps_game.py`. Key components:
- Uses LiteLLM for API calls
- Tracks selections and token probabilities
- Generates visualizations with matplotlib

## Conclusion

GPT-4o demonstrates a strong and statistically significant bias toward selecting "Rock" in Rock-Paper-Scissors games, regardless of how the options are presented. This bias reveals interesting patterns in how large language models make "random" selections and suggests potential biases in training data or model architecture. 

