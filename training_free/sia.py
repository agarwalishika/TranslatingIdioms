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
        {"df": 'Dataset/hindi-english_idioms.csv', "language": "Hindi"},
        {"df": 'Dataset/petci_chinese_english_improved.csv', "language": "Chinese"}
    ]

    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=512
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
            lang = dataset['language']
            df = pd.read_csv(dataset['df'])

            prompt_1 = lambda idiom: f"You are a linguistic researcher on idioms and good at {lang} and English. You'll be provided a {lang} idiom and your task is to:\n1. First provide the definitions of the idiom: {idiom}.\n2. Then generate the three most similar English idioms to the {lang} idiom, and make sure to maintain context and cultural nuances.\n\nFollow these instructions:\n1. If you cannot find three similar English idioms, return as many as you can find.\n2. If no English idiom has the same meaning, only define the {lang} idiom.\n3. For good matches, respond with the English idiom and ensure it is an actual idiom, not a literal translation.\n### Similar English idioms or the meaning in English:"

            inputs = df['src'].apply(lambda x: prompt_1(x))
            outputs = llm.generate(inputs, sampling_params=sampling_params)
            outputs = [o.outputs[0].text.strip() for o in outputs]

            #prompt_2 = lambda output, idiom: f"You are a linguistic researcher on idioms and good at {lang} and English. Choose the best English idiom matching the Chinese idiom and its semantic meaning.\n{lang} idiom: {idiom}"

            # TODO: finish the rest of the methodology
            # Construct selection prompts using the candidate idioms generated in stage 1
            prompt_2 = lambda output, idiom: (
                f"You are a linguistic researcher on idioms and good at {lang} and English.\n"
                f"Below are candidate English idioms generated for a {lang} idiom.\n\n"
                f"{lang} idiom: {idiom}\n"
                f"Candidate English idioms and explanations:\n{output}\n\n"
                f"Task:\n"
                f"1. Select the single English idiom that best matches the semantic meaning and pragmatic usage "
                f"of the {lang} idiom.\n"
                f"2. If none are appropriate, respond with 'NO_MATCH'.\n\n"
                f"### Best English idiom:"
            )

            selection_inputs = [
                prompt_2(out, idiom)
                for out, idiom in zip(outputs, df['src'])
            ]

            selection_outputs = llm.generate(
                selection_inputs,
                sampling_params=sampling_params
            )

            best_matches = [
                o.outputs[0].text.strip()
                for o in selection_outputs
            ]

            # TODO: parse the final output and store the final model predictions in a list of strings
            predicted_outputs = [parse_best_line(t) for t in best_matches]

            # save the predicted output to a csv
            df['predicted'] = predicted_outputs
            df.to_csv(f"outputs/{model['shorthand']}-SIA-{lang}-outputs.csv")
        
        del llm

