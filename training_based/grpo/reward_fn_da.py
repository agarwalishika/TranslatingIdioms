import requests

def format_penalties(solution_str):
    if len(solution_str) <= 1:
        return -5
    
    # -*- coding: utf-8 -*-
    def isEnglish(s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True
    
    if not isEnglish(solution_str):
        return -5
    
    return 0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    SERVER_A = "http://172.22.224.40:5143/da-rewards"

    payload = {
        "completions": [solution_str],
        "true_meaning": [extra_info['true_meaning']],
        "literal_translation": [extra_info['literal_translation']],
    }

    r = requests.post(
        SERVER_A,
        json=payload,
        headers={"X-API-Key": ""},
        timeout=30,
    )
    # r.raise_for_status()
    r.raise_for_status()

    qe_rewards = r.json()["da_rewards"]
    return qe_rewards[0] + format_penalties(solution_str)