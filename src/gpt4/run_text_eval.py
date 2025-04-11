import os
import json
import argparse
import pandas as pd
import openai
from openai_key import api_key
from text_utils import (
    basic_prompt, 
    basic_prompt_es,
    basic_prompt_clean_txt,
    basic_prompt_es_clean_txt,
    basic_parser, 
    get_classification_metrics,
)

client = openai.OpenAI(api_key=api_key)
# Documentation: https://platform.openai.com/docs/guides/text-generation?lang=python

# Prompts
prompt_dict = {
    "basic": basic_prompt,
    "basic_es": basic_prompt_es,
    "basic_clean_text": basic_prompt_clean_txt,
    "basic_es_clean_text": basic_prompt_es_clean_txt
}

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="Path to input file")
parser.add_argument("--result_path", type=str, help="Path to result file")
parser.add_argument("--model_name", type=str, default="gpt-4o", help="GPT model to run")
parser.add_argument("--prompt", type=str, default="basic_es_clean_text", help="Prompt type: basic, basic_es, etc.")
parser.add_argument("--testing", action='store_true', help="Run in testing mode with a sample of 10 rows")

args = parser.parse_args()
print("Arguments:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")

os.makedirs(args.result_path, exist_ok=True)


# load data
df = pd.read_csv(args.input_path, header=0)
if args.testing:
    df = df.sample(n=10)

# Apply prompt template
df["prompts"] = df.apply(prompt_dict[args.prompt], axis=1)
LANG = "es" if "_es" in args.prompt else "en"
if LANG == "en":
    sys_instruct = "You are a pattern-following assistant that can only answer with 'Yes' or 'No'. Your goal is to determine whether a text is sexist."
else:
    sys_instruct = "Eres un asistente que sigue patrones y solo puede responder con 'Sí' o 'No'. Tu objetivo es determinar si un texto es sexista."

# API calls
def query_openai_api(prompt):
    response = client.chat.completions.create(
    model=args.model_name,
    messages=[
            {"role": "developer", "content": sys_instruct},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content
df["output_text"] = df["prompts"].apply(query_openai_api)
df["sexist_pred"] = df["output_text"].apply(lambda x: basic_parser(x, language=LANG))

# Save outputs
df.to_json(f"{args.result_path}/outputs.json", orient="records", indent=4)
df[["id", "sexist", "sexist_pred"]].to_csv(f"{args.result_path}/outputs.csv", index=False)
print("Finished saving outputs.")

# Compute metrics
metrics = get_classification_metrics(df, "sexist", "sexist_pred") 

# Include model name in metrics
metrics.update({
    "model": args.model_name,
})
# Save metrics
with open(f"{args.result_path}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Finished saving results.")
