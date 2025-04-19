import os
from litellm import completion
import json
import pydantic
import enum
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import re
from dotenv import load_dotenv

load_dotenv()


models = ["openai/gpt-4o"]
iterations = 50  # 50 iterations per prompt variant

# Define different prompt variants with different ordering of options
prompt_variants = [
    # Original Rock-first ordering
    "Choose one: rock, paper, or scissors?",
    # Paper-first orderings
    "Choose one: paper, rock, or scissors?",
    # Scissors-first orderings
    "Choose one: scissors, rock, or paper?"
]

# Tracking data structures
# Using a nested dictionary to track by prompt variant
all_choices = {}
all_choice_logprobs = {}
all_top_logprobs_data = {}

for variant in prompt_variants:
    all_choices[variant] = []
    all_choice_logprobs[variant] = {'rock': [], 'paper': [], 'scissors': []}
    all_top_logprobs_data[variant] = []

# Run experiment
for model in models:
    print(f"Running experiment for {model} with {len(prompt_variants)} prompt variants...")
    
    for variant_idx, prompt in enumerate(prompt_variants):
        print(f"\nPrompt variant {variant_idx+1}: \"{prompt}\"")
        
        for i in range(iterations):
            if i % 10 == 0 and i > 0:
                print(f"Completed {i} iterations for prompt variant {variant_idx+1}")
                
            try:
                response = completion(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "I will pick:"}
                    ],
                    temperature=1.0,
                    top_p=1.0,
                    logprobs=True,
                    top_logprobs=3,
                )
                
                # Extract choice and logprobs
                content = response.choices[0].message.content.lower()
                if i < 5:  # Only print first 5 iterations per prompt for readability
                    print(f"Iteration {i+1} response: {content}")
                
                # Determine the choice (rock, paper, or scissors) from the response
                if 'rock' in content:
                    choice = 'rock'
                elif 'paper' in content:
                    choice = 'paper'
                elif 'scissors' in content or 'scissor' in content:
                    choice = 'scissors'
                else:
                    choice = 'other'
                    
                all_choices[prompt].append(choice)
                if i < 5:
                    print(f"Detected choice: {choice}")
                
                # Extract top logprobs for first token (usually the choice)
                first_token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                
                # Use a better method to determine which token corresponds to which choice
                # This addresses the issue where multiple tokens might match 'rock', etc.
                token_data = {'rock': None, 'paper': None, 'scissors': None}
                
                # Debug the token issues
                if i < 5:
                    print("Top logprobs for first token:")
                    for lp in first_token_logprobs:
                        print(f"  Token: '{lp.token}', logprob: {lp.logprob}")
                
                # Process the tokens more carefully
                for lp in first_token_logprobs:
                    token_text = lp.token.lower()
                    
                    # Use regex to match whole word "rock", "paper", or "scissors"
                    if re.search(r'\brock\b', token_text):
                        # Only update if higher probability (less negative logprob) or not yet set
                        if token_data['rock'] is None or lp.logprob > token_data['rock']:
                            token_data['rock'] = lp.logprob
                    elif re.search(r'\bpaper\b', token_text):
                        if token_data['paper'] is None or lp.logprob > token_data['paper']:
                            token_data['paper'] = lp.logprob
                    elif re.search(r'\bsc\b|\bscissors\b', token_text):
                        if token_data['scissors'] is None or lp.logprob > token_data['scissors']:
                            token_data['scissors'] = lp.logprob
                    # Handle partial matches (common with tokenization)
                    elif token_text.startswith('rock'):
                        if token_data['rock'] is None or lp.logprob > token_data['rock']:
                            token_data['rock'] = lp.logprob
                    elif token_text.startswith('paper'):
                        if token_data['paper'] is None or lp.logprob > token_data['paper']:
                            token_data['paper'] = lp.logprob
                    elif token_text.startswith('sc') or token_text.startswith('scissors'):
                        if token_data['scissors'] is None or lp.logprob > token_data['scissors']:
                            token_data['scissors'] = lp.logprob
                
                if i < 5:
                    print(f"Processed token data: {token_data}")
                
                all_top_logprobs_data[prompt].append(token_data)
                
                # Record the probabilities for each option
                if token_data['rock'] is not None:
                    all_choice_logprobs[prompt]['rock'].append(token_data['rock'])
                if token_data['paper'] is not None:
                    all_choice_logprobs[prompt]['paper'].append(token_data['paper'])
                if token_data['scissors'] is not None:
                    all_choice_logprobs[prompt]['scissors'].append(token_data['scissors'])
                
                # Save complete response with prompt information
                if i < 5:  # Only save first 5 responses to reduce file size
                    response_dict = response.to_dict()
                    response_dict['prompt_variant'] = prompt
                    with open('rps_results_order.json', 'a') as f:
                        f.write(json.dumps(response_dict, indent=4) + '\n')
                    
            except Exception as e:
                print(f"Error in iteration {i} for prompt variant {variant_idx+1}: {e}")
                import traceback
                traceback.print_exc()

# Analyze the results for each prompt variant
print("\n\n=== ANALYSIS BY PROMPT VARIANT ===")

# Prepare a figure for visualization
fig, axes = plt.subplots(len(prompt_variants), 2, figsize=(15, 5*len(prompt_variants)))

for idx, (prompt, choices) in enumerate(all_choices.items()):
    print(f"\n\n--- Prompt: \"{prompt}\" ---")
    
    # Count choices
    choice_counts = Counter(choices)
    total = len(choices)
    
    print(f"Total iterations: {total}")
    print("\nChoice distribution:")
    for choice, count in choice_counts.items():
        percentage = (count / total) * 100
        print(f"{choice}: {count} ({percentage:.2f}%)")
    
    # Calculate average log probabilities
    avg_logprobs = {}
    for choice, probs in all_choice_logprobs[prompt].items():
        if probs:
            # Filter out None values
            probs = [p for p in probs if p is not None]
            if probs:
                # Convert log probabilities to probabilities, average, then back to log
                avg_prob = np.mean([np.exp(p) for p in probs])
                avg_logprobs[choice] = np.log(avg_prob)
    
    print("\nAverage logprobs for each choice:")
    for choice, avg_lp in avg_logprobs.items():
        print(f"{choice}: {avg_lp:.4f} (probability: {np.exp(avg_lp):.4f})")
    
    # Plot choice distribution
    ax1 = axes[idx, 0]
    ax1.bar(choice_counts.keys(), choice_counts.values())
    ax1.set_title(f'Choice Distribution - Prompt {idx+1}')
    ax1.set_ylabel('Count')
    
    # Plot average probabilities
    ax2 = axes[idx, 1]
    avg_probs = {k: np.exp(v) for k, v in avg_logprobs.items()}
    ax2.bar(avg_probs.keys(), avg_probs.values())
    ax2.set_title(f'Average Probability - Prompt {idx+1}')
    ax2.set_ylabel('Probability')
    ax2.set_ylim(0, 1.0)  # Set y-axis limit for probabilities

# Add a small caption with the prompts
caption = "Prompt variants (different orderings):\n"
for idx, prompt in enumerate(prompt_variants):
    caption += f"{idx+1}: {prompt}\n"
plt.figtext(0.5, 0.01, caption, ha='center', fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for the caption
plt.savefig('rps_analysis_order_fixed.png')
print("\nAnalysis saved to rps_analysis_order_fixed.png")

# Aggregate analysis across all prompts
all_combined_choices = []
for choices in all_choices.values():
    all_combined_choices.extend(choices)

print("\n\n=== AGGREGATE ANALYSIS ACROSS ALL PROMPTS ===")
overall_counts = Counter(all_combined_choices)
overall_total = len(all_combined_choices)

print(f"Total responses: {overall_total}")
print("\nOverall choice distribution:")
for choice, count in overall_counts.items():
    percentage = (count / overall_total) * 100
    print(f"{choice}: {count} ({percentage:.2f}%)")

# Visualize the overall results
plt.figure(figsize=(10, 6))
plt.bar(overall_counts.keys(), overall_counts.values())
plt.title('Overall Choice Distribution Across Different Option Orderings')
plt.ylabel('Count')
plt.savefig('rps_analysis_order_overall_fixed.png')
print("\nOverall analysis saved to rps_analysis_order_overall_fixed.png") 