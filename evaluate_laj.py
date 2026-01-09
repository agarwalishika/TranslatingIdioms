import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# from comet import download_model, load_from_checkpoint
# import evaluate
# import torch
import pandas as pd
# from rouge_score import rouge_scorer
# from sentence_transformers import SentenceTransformer
# import json
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

def calculate_laj(source, predicted, ground_truth):
    # Absolute Grading: Outputs score of 1 to 5

    instructions = ["Translate the following idiom into English: " + s for s in source]

    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

    feedbacks, scores = judge.absolute_grade(
        instructions=instructions,
        responses=predicted,
        rubric=score_rubric,
        reference_answers=ground_truth
    )

    return list(scores)

def compute_results(input_file):
    df = pd.read_csv(input_file, sep="|")

    if "laj" in list(df.columns) and -1 not in list(df['laj']):
        return

    laj = calculate_laj(list(df['source']), list(df['predicted']), list(df['ground_truth']))
    df['laj'] = laj
    df.to_csv(input_file, sep="|")

from argparse import ArgumentParser
if __name__ == "__main__":
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0", max_num_seqs=16, gpu_memory_utilization=0.8)
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    rubric_data = {
        "criteria": "Evaluate how effectively the translation conveys the meaning of the original idiom in natural English. A good translation should capture the semantic meaning and cultural intent, not necessarily be word-for-word. Focus on meaning equivalence and natural English expression.",
        
        "score1_description": "The translation is missing, nonsensical, or completely unrelated to the source idiom with no discernible connection to the original meaning.",
        
        "score2_description": "The translation attempts to convey meaning but fails significantly. It may be overly literal, making it confusing or unnatural in English, or it misses the core meaning of the idiom entirely.",
        
        "score3_description": "The translation captures the basic meaning of the idiom but is awkward, unnatural, or incomplete. It might be overly literal or miss important nuances and connotations of the original.",
        
        "score4_description": "The translation effectively conveys the meaning of the idiom in natural English with only minor issues. It may have slight awkwardness in phrasing, miss subtle nuances, or use a less common equivalent when a more standard one exists.",
        
        "score5_description": "The translation perfectly captures the meaning, tone, and intent of the original idiom in fluent, natural English. It uses an appropriate English equivalent idiom when one exists, or provides a clear, natural explanation of the concept. Minor differences in filler words or articles that don't affect meaning are acceptable."
    }

    from glob import glob
    files = glob('results/*.csv')

    for file in files:
        compute_results(file)
        print(f'done with {file}')