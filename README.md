VM-Rec
code for paper "VM-Rec: A Variational Mapping Approach for Cold-start User Recommendation"

Requirements
pytorch==1.13.1
recbole==1.1.1
numpy==1.23.5
pandas==2.0.0
tqdm==4.65.0
Usage
python vm_main.py --model BPR --dataset ml-100k
By default, the model is trained on the ml-100k dataset and base recommender is BPR (pre-trained). Data samples and models will be saved in the saved folder.

You can retrain the base recommender by running python run_recbole.py.
