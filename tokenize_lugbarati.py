import os
from transformers import AutoTokenizer
from datasets import load_from_disk

def preprocess_monolingual_lugbarati(examples, tokenizer):
    lugbarati_text = examples["text_transcript"]

    model_inputs = tokenizer(
        lugbarati_text, 
        max_length=128, 
        truncation=True, 
        padding="max_length"
    )
    
    model_inputs["labels"] = [list(ids) for ids in model_inputs["input_ids"]]
    
    return model_inputs

def main():

    dataset_path = "./data/lugbarati_final_build"
    
    if not os.path.exists(dataset_path):
        print(f" Could not find folder '{dataset_path}'")
        return
        
    print(" Loading my full Lugbarati master build from the data folder...")
    raw_dataset = load_from_disk(dataset_path)
    

    model_id = "facebook/nllb-200-distilled-600M"
    print(f" Loading vocabulary mappings from open-source repository: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="lgb_Latn")
    
    print(" Mapping text files into numeric vector arrays...")
    tokenized_dataset = raw_dataset.map(
        lambda x: preprocess_monolingual_lugbarati(x, tokenizer),
        batched=True,
        remove_columns=raw_dataset["train"].column_names
    )

    output_dir = "./data/lugbarati_tokenized_build"
    tokenized_dataset.save_to_disk(output_dir)
    print(f"\n SUCCESS! All {len(tokenized_dataset['train'])} train lines tokenized and ready at: '{output_dir}'")

if __name__ == "__main__":
    main()
