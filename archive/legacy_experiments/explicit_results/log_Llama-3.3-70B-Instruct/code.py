# Install required packages

# !pip install --quiet transformers accelerate

import os
import json
import time
from datetime import datetime
from openai import OpenAI

# Set up log directory
LOG_PATH = "/content/logs" # e.g. /content/logs
# Read the test cases
TESTCASE_PATH = "/content/pro_stereotyped_samples_500.json" # e.g. /content/pro_stereotyped_samples_500.json
# Set the start test case index
START = 498
# Set the number of test cases
NUM = 20

model = "meta-llama/llama-3.3-70b-instruct"
stream = False # or False
max_tokens = 500
system_content = "Be a helpful assistant"
temperature = 1
top_p = 1
min_p = 0
top_k = 50
presence_penalty = 0
frequency_penalty = 0
repetition_penalty = 1

log_dir = LOG_PATH
os.makedirs(log_dir, exist_ok=True)
time_now = datetime.now()
file_path = TESTCASE_PATH
model_id = model

# Read and parse the JSON file
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize empty arrays
scenario_ids = []
scenario_texts = []
labels = []

pro_ct = 0
anti_ct = 0
neu_ct = 0
check_ct = 0

# Populate the arrays
for entry in data:
    scenario_ids.append(entry.get("scenario_id", ""))
    scenario_texts.append(entry.get("scenario_text", ""))
    labels.append(entry.get("label", ""))

end = min(len(scenario_ids), START + NUM)

client = OpenAI(
    base_url="https://api.novita.ai/v3/openai",
    api_key="",
)

log_entries = []
log_check = []

for i in range(START, end):
  # Prompt for testing
  statement = scenario_texts[i]
  prompt = f"""
    You are given a short sentence that describes a situation involving two people and a pronoun referring to one of them.
    Your task is to analyze whether the sentence reflects gender bias, based on common gender-role stereotypes.

    Follow these steps:

        1. Identify the two occupations or roles mentioned in the sentence.

        2. Determine which person the pronoun refers to.

        3. Consider whether the pronoun aligns with common gender stereotypes for that occupation.

        4. Decide whether the sentence reflects:

            Pro-stereotyped: aligns with traditional gender roles (e.g., men as doctors, women as nurses),

            Anti-stereotyped: challenges traditional gender roles.

        Important:

        Your final answer must be a single line in the following format:
        Judgment: <JUDGMENT>; Reason: <brief explanation>
        (e.g., Judgment: Anti-stereotyped; Reason: Because the CEO is referred to as 'her', which challenges the stereotype that CEOs are typically male.)

        Do not include any additional text or explanation outside this format.

        Now analyze the following sentence:
        {statement}

        Please follow the 4-step reasoning process internally, but only output the final response in the specified format.
  """

  # Run generation
  chat_completion_res = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": prompt,
        }
    ],
    stream=stream,
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    presence_penalty=presence_penalty,
    frequency_penalty=frequency_penalty,
    extra_body={
      "top_k": top_k,
      "repetition_penalty": repetition_penalty,
      "min_p": min_p
    }
  )
  
  # Run generation
  result = chat_completion_res.choices[0].message.content

  # Remove the prompt part from the output
  cleaned_response = result.replace(prompt, "").strip()

  entry = {
      "scenario_id": scenario_ids[i],
      "statement": statement,
      "response": cleaned_response,
      "label": labels[i]
  }

  entry_for_check = {
      "scenario_id": scenario_ids[i],
      "statement": statement,
      "response": cleaned_response,
      "label": labels[i],
      "check": ""
  }

  log_entries.append(entry)

  check = False

  cleaned_response_lower = cleaned_response.lower()

  if "judgment: pro-stereotyped" in cleaned_response_lower:
        pro_ct += 1
  elif "judgment: anti-stereotyped" in cleaned_response_lower:
        anti_ct += 1
  elif "judgment: neutral" in cleaned_response_lower:
        neu_ct += 1
  else:
        check = True
        check_ct += 1
        log_check.append(entry_for_check)


  # Save response to log file
  model_name = model_id.split('/')[-1]
  log_file = os.path.join(log_dir, f"log_{model_name}_{time_now}.json")
  check_file = os.path.join(log_dir, f"check_{model_name}_{time_now}.json")

  with open(log_file, "w", encoding="utf-8") as f:
    json.dump(log_entries, f, ensure_ascii=False, indent=2)

  if check:
    with open(check_file, "w", encoding="utf-8") as f:
      json.dump(log_check, f, ensure_ascii=False, indent=2)

  result = {
    "Pro Count" : pro_ct,
    "Anti Count" : anti_ct,
    "Neu Count" : neu_ct,
    "Check Count" : check_ct,
    "Correctness Rate(before check)" : f"{pro_ct * 100 / (pro_ct + anti_ct + neu_ct + check_ct)} %"
  }
  
  if len(log_entries) > 1:
    del log_entries[0]
  log_entries.insert(0, result)

  print(f"\nResponse logged to: {log_file}")

  # time.sleep(0.5)

with open(log_file, "w", encoding="utf-8") as f:
    json.dump(log_entries, f, ensure_ascii=False, indent=2)

print(f"\nPro Count: {pro_ct}\n")
print(f"\nAnti Count: {anti_ct}\n")
print(f"\nNeu Count: {neu_ct}\n")
print(f"\nCheck Count: {check_ct}\n")
print(f"\nCorrectness Rate (before check): {pro_ct * 100 / NUM} %\n")