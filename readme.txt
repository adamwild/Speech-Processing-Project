Projet/
	files/
		<word>/
			<wavefiles>
	output_classifiers/
		<#file>.csv
	vectors/
		<#file>.csv
	data_to_vect.m
	data_to_vect2.m
	info_vectors.txt
	kannumfcc.m (author Olutope Foluso Omogbenigun)
	melbankm.m (author raghu ram)
	run_classifiers.py


The dataset used is:
P. Warden, “Speech commands: A public dataset for singleword
speech recognition.” 2017

Download here:
http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

Setup:
	Place the downloaded dataset in files
	The layout should be like this

	files/
		_background_noise_/
		bed/
		bird/
		...
		validation_list.txt

'files' contains the dataset in wav format

'vectors' contains extracted features from a subset of files from 'files'

'output_classifiers' contains the accuracy of the classifiers trained
from the datasets contained in 'vectors'

data_to_vect.m:
	Computes the features from wav files.
	Store the formed datasets in 'vectors'
	Change line 12 for a different set of words to be used
	Change line 15 for a different number of instances for each word to be used
	Change line 36 for a different feature extraction method

run_classifiers.py:
	Trains MLP and RFC on datasets in 'vectors' 
	Store the results in 'output_classifiers' 
	Change line 26 for a different dataset, 
		if to_analyze = [], all files will be processed, results in 'all.csv'

info_vectors.txt:
	Contains the parameters used to compute the existing files in 'vectors' and 'output_classifiers'