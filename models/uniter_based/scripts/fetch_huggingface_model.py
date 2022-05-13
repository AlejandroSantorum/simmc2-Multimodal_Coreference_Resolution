import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src', type=str)
parser.add_argument('-t', '--type', type=str)
model_checkpoint = parser.parse_args().src
type = parser.parse_args().type
temp = model_checkpoint.split('/')
if len(temp) == 1:
    save_path = temp[0] + '_'
else:
    save_path = '-'.join(model_checkpoint.split('/'))
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForMaskedLM, AutoModelForPreTraining, BertForMaskedLM, AutoModel
AutoTokenizer.from_pretrained(model_checkpoint, force_download=True).save_pretrained(save_path)

print(type)
if type == 'QA':
    AutoModelForQuestionAnswering.from_pretrained(model_checkpoint, force_download=True).save_pretrained(save_path)
elif type == 'Auto':
    AutoModel.from_pretrained(model_checkpoint, force_download=True).save_pretrained(save_path)
elif type == 'MaskLM':
    AutoModelForMaskedLM.from_pretrained(model_checkpoint, force_download=True).save_pretrained(save_path)
elif type == 'BERT':
    AutoModel.from_pretrained(model_checkpoint, force_download=True).save_pretrained(save_path)
elif type == 'SequenceClassification':
    AutoModelForSequenceClassification.from_pretrained(model_checkpoint, force_download=True).save_pretrained(save_path)
elif type == 'TokenClassification':
    AutoModelForTokenClassification.from_pretrained(model_checkpoint, force_download=True).save_pretrained(save_path)
else:
    print("something's wrong?")
import os
os.rename(save_path, save_path[:-1])