import random
import torch
#import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from random import choices
#import matplotlib.pyplot as plt

tqdm.pandas()

from datasets import load_dataset

from transformers import AutoTokenizer, pipeline

from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model,
)