from vllm import LLM, SamplingParams
import pandas as pd
import torch
import re

if __name__ == "__main__":
    models = [
        {"model": "meta-llama/Llama-3.2-1B", "shorthand": "llama-1b"},
        {"model": "Qwen/Qwen2.5-3B", "shorthand": "qwen-3b"},
        
    ]
    datasets = [
        # {"df": 'Dataset/hindi-english_idioms.csv', "language": "Hindi"},
        {"df": 'Dataset/opus_hindi.csv', "language": "Opus_Hindi"},
        # {"df": 'Dataset/petci_chinese_english_improved.csv', "language": "Chinese"}
        {"df": 'Dataset/opus_chinese.csv', "language": "Opus_Chinese"}
    ]

    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=512
    )

    def build_lia_stage1_prompt(lang: str, idiom: str) -> str:
        """
        LIA Stage 1 (Generate): define the source idiom and propose up to 3 real English idioms
        with similar meaning/usage. Allows fewer candidates or definition-only.
        """
        return (
            f"You are a linguistic researcher on idioms and good at {lang} and English.\n"
            f"You will be provided a {lang} idiom.\n\n"
            f"Task:\n"
            f"1) First provide a brief English definition of the {lang} idiom.\n"
            f"2) Then find three most similar English idioms to the {lang} idiom and make sure to maintain context and cultural nuances.\n\n"
            f"Follow these instructions:\n"
            f"- If you cannot find three similar English idioms, return as many as you can.\n"
            f"- If no English idiom has the same meaning, only define the {lang} idiom.\n"
            f"- Do NOT give a literal translation as an idiom.\n\n"
            f"{lang} idiom: {idiom}\n\n"
            f"Output format:\n"
            f"Definition: <one-sentence definition>\n"
            f"Candidates:\n"
            f"1) <english idiom>\n"
            f"2) <english idiom>\n"
            f"3) <english idiom>\n"
        )

    def build_lia_stage2_prompt(lang: str, idiom: str, stage1_output: str) -> str:
        """
        LIA Stage 2 (Select): choose the single best English idiom from the candidates,
        or output NO_MATCH if none fit.
        """
        return (
            f"You are a linguistic researcher on idioms and good at {lang} and English.\n"
            f"Select the single best English idiom that matches the definition of the {lang} idiom.\n\n"
            f"{lang} idiom: {idiom}\n\n"
            f"Stage-1 output (definition + candidates):\n"
            f"{stage1_output}\n\n"
            f"Output format:\n"
            f"Best: <select the most relevant English idiom>\n"
        )

    def parse_best_line(text: str) -> str:
        """
        Extracts the best prediction from the Stage-2 output.
        Falls back gracefully if the model doesn't follow the format.
        """
        t = text.strip()

        # Preferred: line starting with "Best:"
        m = re.search(r"(?im)^\s*Best\s*:\s*(.+?)\s*$", t)
        if m:
            return m.group(1).strip().strip('"').strip("'")

        # Accept exact NO_MATCH anywhere prominent
        m = re.search(r"(?im)\bNO_MATCH\b", t)
        if m:
            return "NO_MATCH"

        # Fallback: take the first non-empty line, stripped
        first_line = next((ln.strip() for ln in t.splitlines() if ln.strip()), "")
        return first_line.strip().strip('"').strip("'") if first_line else "NO_MATCH"

    for model in models:
        llm = LLM(model['model'], tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.7, trust_remote_code=True)

        for dataset in datasets:
            for i in range(32):
                lang = dataset['language']
                df = pd.read_csv(dataset['df'])
                if "hindi" in dataset['df']: df = df[800:]
                elif "chinese" in dataset['df']: df = df[1000:]
                else: 0/0

                # TODO: finish the methodology
                stage1_inputs = df["src"].astype(str).apply(lambda x: build_lia_stage1_prompt(lang, x)).tolist()
                stage1_raw = llm.generate(stage1_inputs, sampling_params=sampling_params)
                stage1_outputs = [o.outputs[0].text.strip() for o in stage1_raw]

                # Stage 2: candidate selection
                stage2_inputs = [
                    build_lia_stage2_prompt(lang, idiom, out)
                    for idiom, out in zip(df["src"].astype(str).tolist(), stage1_outputs)
                ]
                stage2_raw = llm.generate(stage2_inputs, sampling_params=sampling_params)
                stage2_outputs = [o.outputs[0].text.strip() for o in stage2_raw]

                # TODO: parse the final output and store the final model predictions in a list of strings
                predicted_outputs = [parse_best_line(t) for t in stage2_outputs]

                # save the predicted output to a csv
                df['predicted'] = predicted_outputs
                df.to_csv(f"outputs/{model['shorthand']}-LIA-{lang}-outputs_{i}.csv")
        
        del llm

