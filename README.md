<h1>Warning: RMSE scores reported in paper are incorrect</h1>

See [Issue 1](https://github.com/robert-bevan/bglp/issues/1) - a bug in the implementation means RMSE scores are averaged across batches instead of being calculated for the full test set in one go. As a result the reported RMSE metrics are overly optimistic (i.e. lower than they should be). I plan to re-run the experiments and post the updated results here.






<h3>About</h3>
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
