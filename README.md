# Process Extraction From Text for the Dutch Law

This code was created as a part of my Master's thesis project.

## Add data
The data folder should be filled with the law books in txt format.
The dataset.xlsx file should be added to the `model_evaluation_code` folder.

## Overview of notebooks
The `model_code` folder contains all scripts used to make the different flowcharts. 

`event_extraction.py` contains the code of the rule-based algorithm. It runs on one book of law at a time. If you want to change the book of law being processed you should change the data input path in line 2817 and the output path in line 2869. It will store the results of the event extraction and reference resolution in the `extracted_events` folder and the created flowcharts in the `flows` folder.

`flow_chart.ipynb` is used to create a flowchart from the ChatGPT improve flow output. This result will be saved in the `model_code` folder

`LLM_based_flow.ipynb` is used to create the input for the ChatGPT prompts made to improve the rule-based results.

`look_at_parsing_tree.ipynb` is used to look at the parsing tree of a single sentence to analyze it and the effect of the code on the sentence.

The `model_evaluation_code` folder contains the scripts used to analyze the results of the model evaluation.

`summary_statistics.ipynb` is used to calculate summary statistics, test statistics and correlations from the results of the model evaluation.

`Regression models.R` is used to run regression models on the accuracy and time used to answer the different questions in the model evaluation.
