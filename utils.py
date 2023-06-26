import nltk
import json
import transformers 
import torch
import os
import pandas as pd
import numpy as np
import metrics


def clean_text(text):
    cleaned = []
    treebank_detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()
    for t in text:
        t = treebank_detokenizer.detokenize(t.strip().split())
        if len(t.split()) < 10:
            continue
        cleaned.append(t.replace(' .', '.').replace(' @-@ ', '-').replace("\'","'"))
    return cleaned


def load_gpt2_dataset(json_file_name, num_examples=float('inf')):
    texts = []
    for i, line in enumerate(open(json_file_name)):
        if i >= num_examples:
            break
        texts.append(json.loads(line)['text'])
    return texts


def get_mauve_scores_from_files(data_dir):
    all_results = {f: json.load(open(os.path.join(data_dir, f))) for f in os.listdir(data_dir) if 'json' in f}
    df = pd.DataFrame.from_dict(all_results, orient='index').reset_index(names='file')
    def get_adapter_param_from_name(x):
        split =  x[:-5].split('_')
        try:
            param = float(split[-1])
            split = split[:-1]
        except ValueError:
            param = np.nan
        return split[-1], param
    df['adapter'], df['param'] =  zip(*(df['file'].apply(get_adapter_param_from_name)))
    return df


class EtaWarper(transformers.LogitsWarper):
    # Taken from...
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.filter_value = -float("Inf")

    def __call__(self, input_ids, scores) -> torch.FloatTensor:
        probabilities = scores.softmax(dim=-1)
        entropy = torch.distributions.Categorical(probs=(scores).softmax(dim=-1)).entropy()
        epsilon = torch.min(torch.tensor([self.epsilon], device=entropy.device), torch.sqrt(torch.tensor(self.epsilon))*torch.exp(-entropy))
        indices_to_remove = probabilities < epsilon.unsqueeze(dim=1)
        max_word = torch.argmax(scores,dim=-1)
        indices_to_remove[...,max_word.squeeze()] = 0
        new_scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return new_scores

    
def dict_to_df(dictionary):
    all_dfs = []
    for m, dikt in dictionary.items():
        for warper, params in dikt.items():
            warper = warper.__name__ if not isinstance(warper, str) else warper
            for param, results in params.items():
                tmp = {}
                for f in metrics.Metrics._fields:
                    tmp[f] = np.array([getattr(r, f) for r in results], dtype=object)
                all_dfs.append(pd.DataFrame.from_dict(tmp).assign(param=param, adapter=warper, comparison=m))
    return pd.concat(all_dfs).reset_index(drop=True).fillna(value=np.nan)

    
def get_model_outputs(text, model, tokenizer):
    encodings = tokenizer(text, truncation=True, return_offsets_mapping=True)
    tensor_input = torch.tensor([tokenizer.bos_token_id] + encodings['input_ids'], device=model.device)
    with torch.no_grad():
        output = model(tensor_input, labels=tensor_input)
    logits = output['logits'][..., :-1, :].contiguous()
    logits = torch.nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=1)
    labels = tensor_input[..., 1:].contiguous()
    
    neg_log_likelihood = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
    assert torch.isclose(sum(neg_log_likelihood)/len(neg_log_likelihood), output['loss'])
    
    return logits, labels


def get_model_outputs2(text, cur_model, cur_tokenizer, recompute=False, device=None, tensor_dir='data/tensors'):
    encodings = cur_tokenizer(text, truncation=True, return_offsets_mapping=True)
    tensor_input = torch.tensor([cur_tokenizer.bos_token_id] + encodings['input_ids'], device=cur_model.device)
    tensor_path = os.path.join(tensor_dir, str(hash(text)) + '-' + cur_model.config._name_or_path.replace('/','-') + '.pt')
    
    if recompute or not os.path.isfile(tensor_path):
        with torch.no_grad():
            output = cur_model(tensor_input, labels=tensor_input)
        logits = output['logits'][..., :-1, :].contiguous().clone()
        logits = torch.nn.functional.log_softmax(logits.view(-1, logits.size(-1)), dim=1)

        neg_log_likelihood = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), tensor_input[..., 1:].contiguous().view(-1), reduction='none')
        assert torch.isclose(torch.exp(sum(neg_log_likelihood)/len(neg_log_likelihood)), torch.exp(output['loss']))
        torch.save(logits, tensor_path)
    else:
        logits = torch.load(tensor_path)
    if device is not None:
        logits = logits.to(device)
        tensor_input = tensor_input.to(device)
    return logits, tensor_input


