'''
python3 /scratch/project/dd-24-66/it4i-danaesv/sexism_identification_tiktok/conll_sexism/src/claude/run_video_eval.py \
    --input_path /scratch/project/dd-24-66/it4i-danaesv/sexism_identification_tiktok/conll_sexism/corpus/03_Video/video_sexist_only.csv \
    --frame_dir /scratch/project/dd-24-66/it4i-danaesv/sexism_identification_tiktok/conll_sexism/corpus/03_Video/frames \
     --result_path /scratch/project/dd-24-66/it4i-danaesv/sexism_identification_tiktok/conll_sexism/results/video_text/claude/claude_results_video_es_clean \
     --model_name "claude-3-7-sonnet-20250219" \
     --prompt "basic_video_prompt_es" \
     --testing
'''
import os
import json
import argparse
import pandas as pd
import anthropic
import base64
import time
from claude_key import api_key
from text_utils import (
    basic_video_prompt_es,
    basic_parser,
    get_classification_metrics,
)

# Documentation: https://docs.anthropic.com/en/docs/build-with-claude/vision
client = anthropic.Anthropic(api_key=api_key)

prompt_dict = {
    "basic_video_prompt_es": basic_video_prompt_es
}

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="Path to input file")
parser.add_argument("--frame_dir", type=str, help="Path to directory containing extracted frames")
parser.add_argument("--result_path", type=str, help="Path to result file")
parser.add_argument("--model_name", type=str, default="claude-3-7-sonnet-20250219", help="Claude model to run")
parser.add_argument("--max_tokens", type=int, default=10, help="Maximum number of tokens to generate (default: 10)")
parser.add_argument("--prompt", type=str, default="basic_video_prompt_es", help="Prompt type: basic, basic_es, etc.")
parser.add_argument("--testing", action='store_true', help="Run in testing mode with a sample")

args = parser.parse_args()
print("Arguments:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")

os.makedirs(args.result_path, exist_ok=True)

# Load existing results if available
output_file = f"{args.result_path}/outputs.json"
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        existing_results = json.load(f)
    existing_results_df = pd.DataFrame(existing_results)
else:
    existing_results_df = pd.DataFrame()

# Load input data
df = pd.read_csv(args.input_path, header=0)
if args.testing:
    df = df.iloc[:50]  # Process only 2 videos for testing
print(df[["id","text_clean"]])

# Filter out already processed videos
if not existing_results_df.empty:
    df = df[~df["video_name"].isin(existing_results_df["video_name"])]

# Exit early if there's nothing to process
if df.empty:
    print("All videos have already been classified. Exiting.")
    metrics = get_classification_metrics(existing_results_df, "sexist", "sexist_pred")
    metrics.update({"model": args.model_name})
    
    # Save metrics
    with open(f"{args.result_path}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Finished saving results.")
    exit()

# Apply prompt template
df["prompts"] = df.apply(prompt_dict[args.prompt], axis=1)
LANG = "es" if "_es" in args.prompt else "en"
sys_instruct = (
    "You are a video classification assistant. Answer only with 'Yes' or 'No'."
    if LANG == "en"
    else "Eres un asistente de clasificación de videos. Responde solo con 'Sí' o 'No'."
)

# Helper function to encode images as base64
def encode_image(image_path):
    """Encodes an image as base64 and formats it for Claude API."""
    media_type = "image/jpeg"
    with open(image_path, "rb") as image_file:
        base64_image = base64.standard_b64encode(image_file.read()).decode("utf-8")
    return{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64_image,
            },
        }

# Function to retrieve all frames from directory
def get_frames(frame_folder):
    """Retrieve all frames from a folder."""
    if not os.path.exists(frame_folder):
        print("Frames folder doesn't exist")
        return []  # If folder doesn't exist, return empty list

    frames = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".jpg")])
    return frames[:10]  # Ensure max 10 frames per video

# Function to send API request with frames + text prompt
# API call
def query_claude_api(prompt, image_paths):
    time.sleep(1)
    video_name = image_paths[0].split("/")[-1].split("_")[0]
    try:
        message = client.messages.create(
            model=args.model_name,
             max_tokens=args.max_tokens,
            messages=[
                {"role": "user", 
                 "content": [{"type": "text", "text":sys_instruct + prompt}] + [encode_image(img) for img in image_paths]}]
        )
        with open(args.result_path + "/outputs/video_{}.txt".format(video_name), "w") as text_file:
            print(message.content[0].text, file=text_file)
       
        return message.content[0].text
    
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Process each video
df["frame_folder"] = df["video_name"].apply(lambda x: os.path.join(args.frame_dir, x.split("_")[-1]))
df["frames"] = df["frame_folder"].apply(get_frames)
df["output_text"] = df.apply(lambda row: query_claude_api(row["prompts"], row["frames"]), axis=1)
df["sexist_pred"] = df["output_text"].apply(lambda x: basic_parser(x, language=LANG))
#remove path from frames and frame_folder before saving
df["frame_folder"] = df["frame_folder"].apply(lambda x: "/".join(x.split("/")[-2:]))
df["frames"] = df["frames"].apply(lambda x: [z.split("/")[-1] for z in x])

# Append new results to existing results
if not existing_results_df.empty:
    updated_results = pd.concat([existing_results_df, df], ignore_index=True)
else:
    updated_results = df

# Save outputs
updated_results.to_json(output_file, orient="records", indent=4)
updated_results[["id", "sexist", "sexist_pred"]].to_csv(f"{args.result_path}/outputs.csv", index=False)
print("Finished saving outputs.")

# Compute metrics if ground truth labels exist
if "sexist" in df.columns:
    metrics = get_classification_metrics(updated_results, "sexist", "sexist_pred")
    metrics.update({"model": args.model_name})
    
    # Save metrics
    with open(f"{args.result_path}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Finished saving results.")

