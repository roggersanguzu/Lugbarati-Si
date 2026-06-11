import os
import re
import pandas as pd
from datasets import Dataset, DatasetDict

def clean_pure_lugbarati(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
def parse_pure_text_file(file_path):
    lugbarati_sentences = []
    print(f" Processing file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned_line = clean_pure_lugbarati(line)
            if cleaned_line:
                lugbarati_sentences.append(cleaned_line)
    df = pd.DataFrame({
        "text_transcript": lugbarati_sentences,
        "source_file": os.path.basename(file_path)
    })
    print(f"{os.path.basename(file_path)} completed. Total clean lines: {len(df)}")
    return df
def main():
    text_files = ["data1.txt", "data2.txt", "data3.txt", "data4.txt"]
    all_dfs = []
    for file_name in text_files:
        if os.path.exists(file_name):
            df = parse_pure_text_file(file_name)
            if not df.empty:
                all_dfs.append(df)
        else:
            alt_path = os.path.join("data", file_name)
            if os.path.exists(alt_path):
                df = parse_pure_text_file(alt_path)
                if not df.empty:
                    all_dfs.append(df)
            else:
                print(f" Error: File '{file_name}' not found. Check your path!")
    if not all_dfs:
        print(" No text data was found. Check file names.")
        return
    master_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTrue Combined Total Monolingual Lines: {len(master_df)}")
    master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(master_df) * 0.9)
    train_df = master_df.iloc[:split_idx]
    val_df = master_df.iloc[split_idx:]

    hf_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df)
    })
    output_dir = "./lugbarati_final_build"
    hf_dataset.save_to_disk(output_dir)
    print(f"\nSUCCESS! My actual large dataset is saved at: '{output_dir}'")
if __name__ == "__main__":
    main()
