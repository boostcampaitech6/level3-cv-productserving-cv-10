import os,json
import torch
import wandb
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from transformers import(
    Pix2StructForConditionalGeneration, 
    Pix2StructProcessor,
    Trainer,
    TrainingArguments
)
import warnings
warnings.simplefilter("ignore")
model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-infographics-vqa-base").to("cuda")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-infographics-vqa-base")
model.load_state_dict(torch.load('/home/chy/level3-cv-productserving-cv-10/pix2struct_1/checkpoint-48/pytorch_model.bin'))
image = Image.open("./hy_info/task3//images/10065.jpeg")
question = "Which market crash had the lowest impact on the S&P 500, Dot-com crash, Coronavirus crash, or Great recession ?"
inputs = processor(images=image, text=question, return_tensors="pt").to("cuda")
predictions = model.generate(**inputs)
pred = processor.decode(predictions[0], skip_special_tokens=True)

print('Question : ', question,'\nAnswer   : ', pred)