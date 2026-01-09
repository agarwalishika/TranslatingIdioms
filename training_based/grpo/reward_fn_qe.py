import requests
from reward_fn_da import format_penalties

def remove_emojis(text):
    import re


    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text) # no emoji

def compute_pos(data_source, solution_str, ground_truth, extra_info=None):
    SERVER_A = "http://172.22.224.40:5142/qe-rewards"

    payload = {
        "completions": [remove_emojis(solution_str)],
        "mt": [remove_emojis(extra_info['true_meaning'])],
    }

    r = requests.post(
        SERVER_A,
        json=payload,
        headers={"X-API-Key": ""},
        timeout=30,
    )
    # r.raise_for_status()
    r.raise_for_status()
    
    qe_rewards = r.json()["qe_rewards"]
    return qe_rewards[0] + format_penalties(solution_str)

def compute_neg(data_source, solution_str, ground_truth, extra_info=None):
    SERVER_A = "http://172.22.224.40:5142/qe-rewards"

    payload = {
        "completions": [remove_emojis(solution_str)],
        "mt": [remove_emojis(extra_info['literal_translation'])],
    }

    r = requests.post(
        SERVER_A,
        json=payload,
        headers={"X-API-Key": ""},
        timeout=30,
    )

    r.raise_for_status()

    qe_rewards = r.json()["qe_rewards"]
    return -1 * qe_rewards[0] + format_penalties(solution_str)

def compute_constrained(data_source, solution_str, ground_truth, extra_info=None):
    return compute_pos(data_source, solution_str, ground_truth, extra_info) + compute_neg(data_source, solution_str, ground_truth, extra_info) - format_penalties(solution_str)