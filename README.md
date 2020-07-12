BGLP 2020 challenge code accompanying submission titled "Experiments in non-personalized future blood glucose level prediction" by Robert Bevan & Frans Coenen.

<h3>Usage</h3>
<h4>To train a model</h4>
python train_model.py config.json

<h4>To evaluate a model</h4>
python eval_model.py config.json

<h3>Configuration</h3>
The JSON config file (config.json) enables you to adjust the various parameters we experimented with in the paper. Note that the default configuration is the one
we found to be optimal in our experiments.

<h3>Data</h3>
In order to reproduce our results, you'll need access to the OhioT1DM dataset. You can request access here: http://smarthealth.cs.ohio.edu/OhioT1DM-dataset.html
