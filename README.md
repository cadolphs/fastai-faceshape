# fastai-faceshape
Trained a fastai model for face shape detection in a Kaggle Notebook for fast iteration.

Then uploaded the fastai learner to HuggingFace

To use the model in a gradio web interface, just clone the repo, install the requirements (modal), get a HuggingFace token and do 

`MODAL_GPU=ANY modal serve simple_script.py`

to get a web endpoint where you can upload images.

Note: The latest version of gradio has nice features like webcam integration, but, for some reason, was hanging when I tried to run the inference. So I downgraded it to ~ 3.6.
