import os
from vllm import LLM, SamplingParams
import pandas as pd
import torch

models = [
    ## grpo models
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Chinese_llama8b-da"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Chinese_llama8b-qe-cons"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Chinese_llama8b-qe-pos"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Chinese_llama8b-qe-neg"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Chinese_qwen3b-da"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Chinese_qwen3b-qe-cons"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Chinese_qwen3b-qe-pos"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Chinese_qwen3b-qe-neg"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Hindi_llama8b-da"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Hindi_llama8b-qe-cons"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Hindi_llama8b-qe-pos"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Hindi_llama8b-qe-neg"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Hindi_qwen3b-da"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Hindi_qwen3b-qe-cons"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Hindi_qwen3b-qe-pos"},
    {"model": "/path/to/grpo/model", "shorthand": "grpo_Hindi_qwen3b-qe-neg"},
    
    ## translation models
    {"model": "CohereLabs/c4ai-command-r-08-2024", "shorthand": "command_r_32b"},
    {"model": "CohereLabs/c4ai-command-r7b-12-2024", "shorthand": "command_r_7b"},
    
    # ## base models
    {"model": "meta-llama/Llama-3.1-8B", "shorthand": "llama_base"},
    {"model": "Qwen/Qwen2.5-3B", "shorthand": "qwen_base"},

    ## sft models
    {"model": "/path/to/sft/model", "shorthand": "qwen_chinese_sft"},
    {"model": "/path/to/sft/model", "shorthand": "qwen_hindi_sft"},
    {"model": "/path/to/sft/model", "shorthand": "llama_chinese_sft"},
    {"model": "/path/to/sft/model", "shorthand": "llama_hindi_sft"},
    
]

if __name__ == "__main__":
    datasets = [
        {"df": 'Dataset/hindi-english_idioms.csv', "language": "Hindi"},
        {"df": 'Dataset/opus_hindi.csv', "language": "Opus_Hindi"},
        {"df": 'Dataset/petci_chinese_english_improved.csv', "language": "Chinese"},
        {"df": 'Dataset/opus_chinese.csv', "language": "Opus_Chinese"}
    ]

    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=512
    )

    for model in models:
        if "command-r-08" in model['model']:
            llm = LLM(model['model'], tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.8, trust_remote_code=True, max_model_len=80000)
        else:
            llm = LLM(model['model'], tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.8, trust_remote_code=True)
        
        for dataset in datasets:
            for i in range(1):
                lang = dataset['language']
                filename = f"outputs/{model['shorthand']}-{lang}-outputs_{i}.csv"

                prompt = lambda idiom: f"Concisely translate the idiom {idiom} semantically into English: "

                df = pd.read_csv(dataset['df'])
                if "opus" not in dataset['df']:
                    if "hindi" in dataset['df']: df = df[800:]
                    elif "chinese" in dataset['df']: df = df[1000:]
                    else: 0/0

                inputs = df['src'].apply(lambda x: prompt(x))
                outputs = llm.generate(inputs, sampling_params=sampling_params)
                outputs = [o.outputs[0].text.strip().replace("\n", " ")[:100] for o in outputs]
            
                df['predicted'] = outputs
                df.to_csv(filename, sep="|")
            
        del llm