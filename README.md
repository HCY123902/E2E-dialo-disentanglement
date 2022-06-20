# Conversation Disentanglement with Bi-Level Contrastive Learning

## Environment

The model is run with Python

## Dataset

The Dataset needs to be placed in the ``dataset/`` folder.

The format of each sample is as:

	[["speaker": "xxx", "utterance": "xxx", "label": "x"], 
	 ["speaker": "xxx", "utterance": "xxx", "label": "x"] ... ]

For each utterance, "speaker" indicates the speaker of the utterance (which is not used in this work), and "utterance" is the content of the current utterance. "label" indicates which session the current utterance belongs to.

To run with the Ubuntu IRC dataset, set ``dataset = 'irc'`` in ``constant.py``. To run with the Movie Script dataset, set ``dataset = 'movie'`` in ``constant.py``

## Code

The code for our proposed end-to-end framework is contained in the ``code/`` folder.

Before you run the code, you have to specify some hyperparameters and some file paths in the file ``constant.py``. Additionally, you will need to create a folder ``./glove`` and place ``glove.840B.300d.txt`` in it.

You can train the model by the following command:

	python main.py --mode train --save_input --device [cpu or gpu index]

After training the model, the test command is:

	python main.py --mode test --save_input --model_path [path to the model you want to test] --device [cpu or gpu index]

