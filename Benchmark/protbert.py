# <h3> Benchmark ProtBert Model using GPU or CPU <h3>

# <b>1. Load necessry libraries including huggingface transformers<b>

# !pip install -q transformers

import torch
from transformers import BertModel
import time
from datetime import timedelta
import requests
import argparse
import os
from tqdm.auto import tqdm
import sys

def main(args):
    os.makedirs(args.modelFolderPath, exist_ok=True)

    def download_file(url, filename):
      response = requests.get(url, stream=True)
      with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                        total=int(response.headers.get('content-length', 0)),
                        desc=filename) as fout:
          for chunk in response.iter_content(chunk_size=4096):
              fout.write(chunk)

    if not os.path.exists(args.modelFilePath):
        download_file(args.modelUrl, args.modelFilePath)

    if not os.path.exists(args.configFilePath):
        download_file(args.configUrl, args.configFilePath)

    if not os.path.exists(args.vocabFilePath):
        download_file(args.vocabUrl, args.vocabFilePath)

    # <b>4. Load ProtBert Model<b>

    model = BertModel.from_pretrained(args.modelFolderPath)

    # <b>5. Load the model into the GPU if avilabile and switch to inference mode<b>

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model = model.eval()

    # <b>6. Benchmark Configuration<b>

    batch_size = 128

    min_sequence_length = 32
    sequence_length_iteration = 6

    iterations = 10

    # <b>7. Start Benchmarking<b>

    device_name = torch.cuda.get_device_name(device.index) if device.type == 'cuda' else 'CPU'

    with torch.no_grad():
        print((' Benchmarking using ' + device_name + ' ').center(80, '*'))
        print(' Start '.center(80, '*'))
        for sequence_length_power in range(sequence_length_iteration):
            sequence_length = min_sequence_length * (2**sequence_length_power)
            start = time.time()
            for i in range(iterations):
                input_ids = torch.randint(1, 20, (batch_size, sequence_length)).to(device)
                results = model(input_ids)[0].cpu().numpy()
            end = time.time()
            ms_per_protein = (end-start) / (iterations*batch_size)
            print('Sequence Length: %4d \t Batch Size: %4d \t Ms per protein %4.2f' %(sequence_length, batch_size, ms_per_protein))
        print(' Finished '.center(80, '*'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # <b>2. Set the url location of ProtBert and the vocabulary file<b>
    parser.add_argument('--modelUrl', default='https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1')
    parser.add_argument('--configUrl', default='https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1')
    parser.add_argument('--vocabUrl', default='https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1')

    # <b>3. Download ProtBert models and vocabulary files<b>
    parser.add_argument('--downloadFolderPath', default='models/ProtBert/')
    parser.add_argument('--modelFolderPath', default='models/ProtBert/')
    parser.add_argument('--modelFile', default='pytorch_model.bin')
    parser.add_argument('--configFile', default='config.json')
    parser.add_argument('--vocabFile', default='vocab.txt')

    args = parser.parse_args([])
    args.modelFilePath = os.path.join(args.modelFolderPath, args.modelFile)
    args.configFilePath = os.path.join(args.modelFolderPath, args.configFile)
    args.vocabFilePath = os.path.join(args.modelFolderPath, args.vocabFile)
    print(vars(args))

    main(args)
