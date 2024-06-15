import os
import json
import datetime
from email.utils import parseaddr

import gradio as gr
import pandas as pd
import numpy as np

from datasets import load_dataset
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import HfApi, snapshot_download


def model_hyperlink(link, model_name):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'


TOKEN = "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"
OWNER = "autogenCTF"
RESULTS_DATASET = f"{OWNER}/test_result"
all_version = ['20240602']
dataset_version = all_version[0]
val_or_test = 'validation'

data = load_dataset(
    RESULTS_DATASET,
    dataset_version,
    token=TOKEN,
    download_mode="force_redownload",
)

filtered_dataset = data.filter(lambda example: example['model'] not in ['aa'])
split = ['test', 'validation']

for item in filtered_dataset['validation']:
    print(item)

# print(filtered_dataset)
# filtered_dataset.push_to_hub(RESULTS_DATASET, config_name=dataset_version, token=TOKEN)
