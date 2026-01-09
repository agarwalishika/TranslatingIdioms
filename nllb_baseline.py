import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import csv
import gc
from vllm import LLM, SamplingParams
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

if __name__ == "__main__":
    datasets = [
        # {"df": 'Dataset/hindi-english_idioms.csv', "language": "Hindi", "lang_code": "hin_Deva"},
        {"df": 'Dataset/opus_hindi.csv', "language": "Opus_Hindi", "lang_code": "hin_Deva"},
        # {"df": 'Dataset/petci_chinese_english_improvOd.csv', "language": "Chinese", "lang_code": "zho_Hant"},
        {"df": 'Dataset/opus_chinese.csv', "language": "Opus_Chinese", "lang_code": "zho_Hant"},
    ]

    models = [
        "facebook/nllb-200-distilled-1.3B",
        "facebook/nllb-200-3.3B"
    ]

    for model in models:
        for dataset in datasets:
            df = pd.read_csv(dataset['df'])
            if "Hindi" == dataset['language']: df = df[800:]
            elif "Chinese" == dataset['language']: df = df[1000:]
            # else: 0/0

            inf_pipeline = pipeline(task="translation", model=model, src_lang=dataset['lang_code'], tgt_lang="eng_Latn", dtype=torch.float16, device=0)

            inputs = list(df['src'])
            for iter in range(1):
                outputs = []
                bs = 8
                for i in tqdm(range(0, len(inputs), bs)):
                    temp = inf_pipeline(inputs[i:i+bs])
                    outputs.extend([o['translation_text'] for o in temp])
            
                lang = dataset['language']
                df['predicted'] = outputs
                df.to_csv(f"outputs/{model.split('/')[-1]}-{lang}-outputs_{iter}.csv", sep="|")
            del inf_pipeline
        
            