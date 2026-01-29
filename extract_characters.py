from datasets import load_dataset
import pandas as pd

dataset_hf = load_dataset(
        "csv",
        data_files="/mnt/c/Files/Code/Benchmark/datasets/ScienceAgentBench/ScienceAgentBench.csv",
        split="train"
    )

df = pd.read_json("claude-code-eval.jsonl", lines=True)

with open("results.txt", "w", encoding="utf-8") as f:
    for index, score in df.iterrows():
        example = dataset_hf[index]
        alike = score["codebert_score"]
        f.write(f"{index}\t{example['task_inst']}\t{alike}\n")

