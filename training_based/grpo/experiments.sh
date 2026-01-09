#### HIGH PRIORITY TRAINING RUNS
# Hindi / Qwen
source run_grpo_verl.sh "Hindi" "Qwen/Qwen2.5-3B" "qwen3b-qe-pos" "reward_fn_qe.py" "compute_pos"
source run_grpo_verl.sh "Hindi" "Qwen/Qwen2.5-3B" "qwen3b-da" "reward_fn_da.py" "compute_score"

# Hindi / Llama
source run_grpo_verl.sh "Hindi" "meta-llama/Llama-3.1-8B" "llama8b-qe-pos" "reward_fn_qe.py" "compute_pos"
source run_grpo_verl.sh "Hindi" "meta-llama/Llama-3.1-8B" "llama8b-da" "reward_fn_da.py" "compute_score"

# Chinese / Qwen
source run_grpo_verl.sh "Chinese" "Qwen/Qwen2.5-3B" "qwen3b-qe-pos" "reward_fn_qe.py" "compute_pos"
source run_grpo_verl.sh "Chinese" "Qwen/Qwen2.5-3B" "qwen3b-da" "reward_fn_da.py" "compute_score"

# Chinese / Llama
source run_grpo_verl.sh "Chinese" "meta-llama/Llama-3.1-8B" "llama8b-qe-pos" "reward_fn_qe.py" "compute_pos"
source run_grpo_verl.sh "Chinese" "meta-llama/Llama-3.1-8B" "llama8b-da" "reward_fn_da.py" "compute_score"


# NEG RUNS
source run_grpo_verl.sh "Hindi" "Qwen/Qwen2.5-3B" "qwen3b-qe-neg" "reward_fn_qe.py" "compute_neg"
source run_grpo_verl.sh "Hindi" "meta-llama/Llama-3.1-8B" "llama8b-qe-neg" "reward_fn_qe.py" "compute_neg"
source run_grpo_verl.sh "Chinese" "Qwen/Qwen2.5-3B" "qwen3b-qe-neg" "reward_fn_qe.py" "compute_neg"
source run_grpo_verl.sh "Chinese" "meta-llama/Llama-3.1-8B" "llama8b-qe-neg" "reward_fn_qe.py" "compute_neg"

# CONS RUNS
source run_grpo_verl.sh "Hindi" "Qwen/Qwen2.5-3B" "qwen3b-qe-cons" "reward_fn_qe.py" "compute_constrained"
source run_grpo_verl.sh "Hindi" "meta-llama/Llama-3.1-8B" "llama8b-qe-cons" "reward_fn_qe.py" "compute_constrained"
source run_grpo_verl.sh "Chinese" "Qwen/Qwen2.5-3B" "qwen3b-qe-cons" "reward_fn_qe.py" "compute_constrained"
source run_grpo_verl.sh "Chinese" "meta-llama/Llama-3.1-8B" "llama8b-qe-cons" "reward_fn_qe.py" "compute_constrained"