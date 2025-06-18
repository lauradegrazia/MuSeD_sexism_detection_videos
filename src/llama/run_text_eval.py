import os
import json
import psutil
import argparse

import pandas as pd
import torch
from vllm import LLM, SamplingParams

from text_utils import (basic_prompt, 
                        basic_prompt_es,
                        basic_prompt_clean_txt,
                        basic_prompt_es_clean_txt,
                        basic_parser, 
                        get_classification_metrics,
                        compute_metrics_stats
                        )




# models

MODELS_PATH = "/path/to/models/"
models = {"llama_70B":"models--meta-llama--Llama-3.1-70B-Instruct/snapshots/945c8663693130f8be2ee66210e062158b2a9693",
          "llama_8B": "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
          }
# prompts
prompt_dict = {
    "basic": basic_prompt,
    "basic_es": basic_prompt_es,
    "basic_clean_text": basic_prompt_clean_txt,
    "basic_es_clean_text": basic_prompt_es_clean_txt
}

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="Path to input file")
parser.add_argument("--result_path", type=str, help="Path to result file")
parser.add_argument("--model_name", type=str, help="model to run")
parser.add_argument("--prompt", type=str, default="basic", help="prompt in: basic, basic_es, basic_clean_text, basic_es_clean_text")
parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling probability (default: 1.0)")
parser.add_argument("--top_k", type=int, default=-1, help="Top-K sampling parameter (default: -1, means disabled)")
parser.add_argument("--max_tokens", type=int, default=10, help="Maximum number of tokens to generate (default: 10)")
parser.add_argument("--max_model_len", type=int, default=4096, help="Maximum model length (default: 4096)")
parser.add_argument("--testing",  action='store_true', help="testing mode sample of 10")


args = parser.parse_args()
print("Arguments:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")

os.makedirs(args.result_path, exist_ok=True)

mem = psutil.virtual_memory()
available_memory = mem.available
print(f"Available memory: {available_memory / (1024 * 1024 * 1024):.2f} GB")
print("Num GPUs: ", torch.cuda.device_count())
LLAMA_PATH = MODELS_PATH + models[args.model_name]
print("\nLLAMA_PATH", LLAMA_PATH)

TEMPERATURE = args.temperature
TOP_P = args.top_p
TOP_K = args.top_k
MAX_TOKENS = args.max_tokens
MAX_MODEL_LEN = args.max_model_len
LANG = "es" if "_es" in args.prompt else "en"

model = LLM(
    model=LLAMA_PATH,
    tokenizer=LLAMA_PATH,
    trust_remote_code=False,
    tensor_parallel_size=torch.cuda.device_count(),
    gpu_memory_utilization=0.90,
    swap_space=0,
    dtype="bfloat16",
    max_model_len=MAX_MODEL_LEN,
    max_logprobs=None,
    enforce_eager=True
)

sampling_config = SamplingParams(
    temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K, max_tokens=MAX_TOKENS
)

df = pd.read_csv(args.input_path, header=0)
if args.testing:
    df = df.sample(n=10)


df["prompts"] = df.apply(prompt_dict[args.prompt], axis=1)
print(df["prompts"].iloc[0])
prompts = df["prompts"].tolist()
for n in range(3):
    outputs = model.generate(prompts, sampling_config, use_tqdm=True)
    outputs_text = [sample.outputs[0].text.strip() for sample in outputs]
    df["output_text_{}".format(n+1)] = outputs_text
    df["sexist_pred_{}".format(n+1)] = [basic_parser(o,language=LANG) for o in outputs_text]

# save outputs
df.to_json(f"{args.result_path}/outputs.json", orient="records")
df[["id","sexist", "sexist_pred_1","sexist_pred_2","sexist_pred_3"]].to_csv(f"{args.result_path}/outputs.csv",index=False)
print("Finished saving outputs.")

metrics_1 = get_classification_metrics(df, "sexist", "sexist_pred_1") 
metrics_2 = get_classification_metrics(df, "sexist", "sexist_pred_2") 
metrics_3 = get_classification_metrics(df, "sexist", "sexist_pred_3") 
metrics = compute_metrics_stats([metrics_1,metrics_2,metrics_3])
# average metrics 

# include params
metrics["model"] = args.model_name
metrics["temperature"] = TEMPERATURE
metrics["TOP_P"] = TOP_P
metrics["TOP_K"] = TOP_K
metrics["MAX_TOKENS"] = MAX_TOKENS
metrics["MAX_MODEL_LEN"] = MAX_MODEL_LEN


with open(f"{args.result_path}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("Finished saving results.")

