# Beware of Sharks!

This is a project repo for Valohai blog post using Huggingface transformers. 

There are the python scripts

1. fecth_data.py #where you fetch the data 
2. pre_process.py # for filtering out stuff
3. prepare_text.py # splitting the data and further preparing
4. fine_tune.py # using transformers for tokenizing and Distilbert model fine-tuning

You can create the valohai.yaml file with [valohai-utils sdk](https://github.com/valohai/valohai-utils). You need to define each step like this ```vh yaml step fecth_data.py``` and then after the pipeline is created with ```vh yaml pipeline create_pipeline.py```.

Requirements are naturally described in requirements.txt file.
