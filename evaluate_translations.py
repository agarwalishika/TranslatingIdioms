import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
from comet import download_model, load_from_checkpoint
import evaluate
import torch
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import json
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

def process(file):
    df = pd.read_csv(file, sep="|")

    df['predicted'] = df['predicted'].fillna("")
    source = list(df['src'])
    predicted = list(df['predicted'])
    ground_truth = list(df['true_meaning'])
    return source, predicted, ground_truth

def calculate_da(source, predicted, ground_truth):
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    data = []
    for s, p, g in zip(source, predicted, ground_truth):
        data.append({
            "src": s,
            "mt": p,
            "ref": g
        })
    
    model_output = model.predict(data, batch_size=8, gpus=1)
    
    del model
    return model_output['scores']

def calculate_qe(source, predicted, ground_truth):
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    data = []
    for s, p, g in zip(source, predicted, ground_truth):
        data.append({
            "src": s,
            "mt": p
        })
    
    model_output = model.predict(data, batch_size=8, gpus=1)
    
    del model
    return model_output['scores']

def calculate_rouge(source, predicted, ground_truth):
    n = 3
    def char_ngrams(text, n):
        text = text.strip()
        if len(text) < n:
            return text
        return " ".join(text[i:i+n] for i in range(len(text) - n + 1))

    scorer = rouge_scorer.RougeScorer([f"rouge{n}"], use_stemmer=False)

    scores = []
    for pred, gt in zip(predicted, ground_truth):
        pred_ngrams = char_ngrams(pred, n)
        gt_ngrams = char_ngrams(gt, n)
        s = scorer.score(gt_ngrams, pred_ngrams)[f"rouge{n}"].fmeasure
        scores.append(s)

    del scorer
    return scores

def calculate_embed_distance(source, predicted, ground_truth):
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    predicted_embeddings = model.encode(predicted)
    ground_truth_embeddings = model.encode(ground_truth)

    similarities = model.similarity(predicted_embeddings, ground_truth_embeddings)

    del model
    return similarities.diag().tolist()


def calculate_laj(source, predicted, ground_truth):
    # Absolute Grading: Outputs score of 1 to 5
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0", max_num_seqs=16, gpu_memory_utilization=0.6)
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    rubric_data = {
        "criteria": "Evaluate how effectively the translation conveys the meaning of the original idiom in natural English. A good translation should capture the semantic meaning and cultural intent, not necessarily be word-for-word. Focus on meaning equivalence and natural English expression.",
        "score1_description": "The translation is missing, nonsensical, or completely unrelated to the source idiom with no discernible connection to the original meaning.",
        "score2_description": "The translation attempts to convey meaning but fails significantly. It may be overly literal, making it confusing or unnatural in English, or it misses the core meaning of the idiom entirely.",
        "score3_description": "The translation captures the basic meaning of the idiom but is awkward, unnatural, or incomplete. It might be overly literal or miss important nuances and connotations of the original.",
        "score4_description": "The translation effectively conveys the meaning of the idiom in natural English with only minor issues. It may have slight awkwardness in phrasing, miss subtle nuances, or use a less common equivalent when a more standard one exists.",
        "score5_description": "The translation perfectly captures the meaning, tone, and intent of the original idiom in fluent, natural English. It uses an appropriate English equivalent idiom when one exists, or provides a clear, natural explanation of the concept. Minor differences in filler words or articles that don't affect meaning are acceptable."
    }

    instructions = ["Translate the following idiom into English: " + s for s in source]

    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

    feedbacks, scores = judge.absolute_grade(
        instructions=instructions,
        responses=predicted,
        rubric=score_rubric,
        reference_answers=ground_truth
    )

    del model, judge
    return list(scores)

def compute_results(input_file, output_file):
    source, predicted, ground_truth = process(input_file)
    da = calculate_da(source, predicted, ground_truth)
    qe = calculate_qe(source, predicted, ground_truth)
    rouge = calculate_rouge(source, predicted, ground_truth)
    embed_distance = calculate_embed_distance(source, predicted, ground_truth)
    laj = calculate_laj(source, predicted, ground_truth)

    df = pd.DataFrame({
        "source": source,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "da": da,
        "qe": qe,
        "rouge": rouge,
        "embed_distance": embed_distance,
        "laj": laj
    })

    df.to_csv(output_file, sep="|")

from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, default="outputs/grpo_Chinese_llama1b-da-Chinese-outputs_0.csv")
    args = parser.parse_args()

    file = args.file
    output_file = file.replace('outputs', 'results')

    if not os.path.exists(output_file):
        compute_results(file, output_file)
        
    print(f'done with {file}')