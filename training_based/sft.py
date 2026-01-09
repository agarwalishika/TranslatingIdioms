from transformers import TrainingArguments
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset

def parse_df(idiom_df: pd.DataFrame, language, length):
    dataset = []
    for i, row in idiom_df.iterrows():
        if len(dataset) >= length:
            break
        dataset.append({
            "source": "GPT4LLM",
            "messages": [{
                "content": f"Given the following idiom in {language}, translate it semantically into English.\nIdiom: " + row['src'] + "\n### Semantic Translation: ",
                "role": "user"
            },
            {
                "content": row['true_meaning'],
                "role": "assistant"
            }],
            "num_turns": 2
        })
    return Dataset.from_list(dataset)

hindi_idioms = parse_df(pd.read_csv('../Dataset/hindi-english_idioms.csv'), "Hindi", 800)
chinese_idioms = parse_df(pd.read_csv('../Dataset/petci_chinese_english_improved.csv'), "Chinese", 1000)


##################################################### TRAINING #####################################################

#---------------------------------------------------- QWEN ----------------------------------------------------#
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-3B",
    train_dataset=hindi_idioms,
    args=TrainingArguments(
        output_dir="/path/to/sft/model"
    )
)
trainer.train()

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-3B",
    train_dataset=chinese_idioms,
    args=TrainingArguments(
        output_dir="/path/to/sft/model"
    )
)
trainer.train()

# #---------------------------------------------------- LLAMA ----------------------------------------------------#
from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
)

tokenizer.chat_template = """
{% for message in messages %}
{% if message['role'] == 'system' %}
<|system|>
{{ message['content'] }}
{% elif message['role'] == 'user' %}
<|user|>
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}
<|assistant|>
{{ message['content'] }}
{% endif %}
{% endfor %}
<|assistant|>
"""

trainer = SFTTrainer(
    model=model_name,
    processing_class=tokenizer,
    train_dataset=hindi_idioms,
    args=TrainingArguments(
        output_dir="/path/to/sft/model"
    )
)
trainer.train()

trainer = SFTTrainer(
    model=model_name,
    processing_class=tokenizer,
    train_dataset=chinese_idioms,
    args=TrainingArguments(
        output_dir="/path/to/sft/model"
    )
)
trainer.train()