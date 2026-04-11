import os
from huggingface_hub import hf_hub_download
import pandas as pd
from datasets import Dataset

# 数据集配置
dataset = load_dataset("Amazon-Reviews-2023")