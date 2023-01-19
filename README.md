# MTA Inference
Code for testing and running Bayesian inference on MTA data.

To set up and run constantly-running inference:

- Create and activate virtual environment:
```
python -m venv env
source env/bin/activate
```
- Install packages:
```
pip install -r requirements.txt
```
- Run inference:
```
python control_inference_sockeye_experiments.py --config /<<file_path>>/example.json --database <<database_name>> --run_name <<run_name>> --dataset_name <<dataset_name>>
```

