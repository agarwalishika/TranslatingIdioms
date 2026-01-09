To train models with GRPO:

1. Process the datasets using `process_translation_ds.py'
2. Configure the following environment variables: `HOSTNAME` (for the server IP address), `PORT_DA` (for the port number for the DA model), and `PORT_QE` (for the port number for the QE model)
3. Run `service_da.py` and `service_qe.py` so we can make API calls to the MTQE models (our lab servers cannot train RL models _and_ host MTQE reward models on the same server, so we set up the RL training on one server and the MTQE models on the other, and they talk to each other vai API calls.)
4. Use the `experiments.sh` file to set up experiments, that will eventually call `run_grpo_verl.sh` where the verl command is run.

As usual, please reach out via email or Github issue for any questions/concerns!