import itertools
import json
import dotenv
from enum import Enum
from typing import List, Dict
from litellm import completion
import pydantic
import uuid

dotenv.load_dotenv()

# Constants
MODEL_NAME = "openai/gpt-4o"
ITERATIONS_PER_SCENARIO = 100
RESULTS_FILE = "rps_results_order.json"
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 100
STOP_SEQUENCE = None
LOGPROBS_TOP_ENTRIES = 5

class Choice(Enum):
    ROCK = "rock"
    PAPER = "paper"
    SCISSORS = "scissors"
    def __str__(self):
        return self.value

class Scenario(pydantic.BaseModel):
    prompt_order: List[Choice]

def build_messages(scenario: Scenario) -> List[Dict[str, str]]:
    session_id = str(uuid.uuid4())
    prompt = f"Session ID: {session_id}\nChoose one: {scenario.prompt_order[0].value}, {scenario.prompt_order[1].value}, or {scenario.prompt_order[2].value}? Respond only with your choice."
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "I pick:\n"}
    ]

all_choices = list(Choice)
all_orderings = list(itertools.permutations(all_choices))
all_scenarios = [
    Scenario(prompt_order=list(order))
    for order in all_orderings
]

results = []
for scenario in all_scenarios:
    print(f"Running scenario: prompt order {scenario.prompt_order}")
    for i in range(ITERATIONS_PER_SCENARIO):
        if i % 10 == 0 and i > 0:
            print(f"Completed {i} iterations for scenario {scenario.prompt_order}")
        messages = build_messages(scenario)
        response = completion(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            logprobs=True,
            top_logprobs=LOGPROBS_TOP_ENTRIES,
            max_tokens=MAX_TOKENS,
            stop=STOP_SEQUENCE,
        )
        first_token_logprobs = response.choices[0].logprobs["content"][0]["top_logprobs"]
        results.append({
            "prompt_order": [c.value for c in scenario.prompt_order],
            "response": response.choices[0].message.content.lower(),
            "logprobs": [
                {"token": entry.token, "logprob": entry.logprob}
                for entry in first_token_logprobs
            ],
        })

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=4)