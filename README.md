# A Rising Tide Lifts All Boats: MTQE Rewards for Idioms Improve General Translation Quality

Here, we include all our code for our paper.

## Directory Structure
### Training-Based methods:
You can find the SFT and GRPO training code in `training_based/`, along with instructions on how to run the GRPO models.

### Training-Free methods:
You can find LIA and our TrainingFree prompting method in `training_free/`.

### Model baselines
You can find the NLLB baseline in `nllb_baseline.py` and Command-R in `model_inference.py`.

## To Reproduce Results
First, we need to generate model translations for each baseline (which are all stored in `outputs/`):
1. Run the Training-Free methods to get the model translations for LIA and TrainingFree
2. Train the Training-Based models
3. Change the SFT and GRPO model paths in `model_inference.py`, and run that to get model translations for SFT, GRPO, Base, and Command-R models.
4. Run `nllb_baseline.py` to get model translations for the NLLB models.

---

Now, we can evaluate the translations -- these will be stored in `results/`. All the evaluation metrics are in `evaluate_translations.py`. You can use the following code to print a bash script to evaluate all the model translations:

```
from glob import glob
output_files = glob("outputs/*.csv")
for output_file in output_files:
    with open('evaluate_translations.sh', 'a+') as f:
        f.write(f"py evaluate_translations.py --file \"{output_file}\"\n")
```

---

Next, we can plot the results. It's _very important_ to sort the `results/` folder into 8 subfolders:
- `chinese_eval` for all evaluations on the Chinese idioms dataset (should have 20 files)
- `hindi_eval` for all evaluations on the Hindi idioms dataset (should have 20 files)
- `opus_chinese_eval` for all evaluations on the Chinese non-idiomatic dataset (should have 20 files)
- `opus_hindi_eval` for all evaluations on the Hindi non-idiomatic dataset (should have 20 files)
- `transfer_eval_hi2zh` for all transfer evaluations, of models trained on Hindi and evaluated on Chinese idioms (should have 10 files)
- `transfer_eval_zh2hi` for all transfer evaluations, of models trained on Chinese and evaluated on Hindi idioms (should have 10 files)
- `opus_transfer_hi2zh` for all transfer evaluations, of models trained on Hindi and evaluated on Chinese non-idiomatic data (should have 10 files)
- `opus_transfer_zh2hi` for all transfer evaluations, of models trained on Chinese and evaluated on Hindi non-idiomatic data (should have 10 files)

Now, we can use `plot_results.py` to plot all the results.

## Contact
If there are any questions/issues, please feel free to raise a Github issue, or reach out to Ishika Agarwal (ishikaa2 AT illinois DOT edu)