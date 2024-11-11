from torch.utils.data.dataloader import DataLoader,RandomSampler, SequentialSampler
from datasets import Dataset, DatasetDict
from pathlib import Path
import os
from tqdm import tqdm
import umap
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from source.model import ModelWiseCNNClassifier, ModelWiseTransformerClassifier, TransformerOnlyClassifier
ckpt_path='/mnt/chenxi/project/SeqXGPT/SeqXGPT/SeqXGPT/source/linear_en.pt'

class DataManager:
	
    def __init__(self) -> None:
        
        self.max_len=512
        self.label_pad_idx=-1
        self.human_label='human'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.id2label,self.label2id,self.en_labels=self.get_label2id()
    def get_label2id(self):
        en_labels = {
        'gpt2': 0,
        #'gptneo': 1,
        #'gptj': 2,
        'llama': 2,
        'gpt3': 3,
        'human':4,
        'gpt4': 5,
        'gpt4A': 6,
        'gpt3A':7,
        'qwen':8
        }
    # en_labels = {'AI':0, 'human':1}

        id2label = self.construct_bmes_labels(en_labels)
        label2id = {v: k for k, v in id2label.items()}
        return id2label,label2id,en_labels
    def construct_bmes_labels(self,labels):
        prefix = ['B-', 'M-', 'E-', 'S-']
        id2label = {}
        counter = 0

        for label, id in labels.items():
            for pre in prefix:
                id2label[counter] = pre + label
                counter += 1
    
        return id2label
    def _split_en_sentence(self, sentence, use_sp=False):
        import re
        pattern = re.compile(r'\S+|\s')
        words = pattern.findall(sentence)
        if use_sp:
            words = ["▁" if item == " " else item for item in words]
        return words

    def sequence_labels_to_ids(self, seq_len, label):
        prefix = ['B-', 'M-', 'E-', 'S-']
        if seq_len <= 0:
            return None
        elif seq_len == 1:
            label = 'S-' + label
            return torch.tensor([self.label2id[label]], dtype=torch.long)
        else:
            ids = []
            ids.append(self.label2id['B-'+label])
            ids.extend([self.label2id['M-'+label]] * (seq_len - 2))
            ids.append(self.label2id['E-'+label])
            return torch.tensor(ids, dtype=torch.long)

    def process_and_convert_to_tensor(self, data):
        """ here, data is features. """
        max_len = self.max_len
        # data shape: [B, S, E]

        feat_dim = len(data[0][0])
        padded_data = [  # [[0] * feat_dim] + 
            seq + [[0] * feat_dim] * (max_len - len(seq)) for seq in data
        ]
        padded_data = [seq[:max_len] for seq in padded_data]

        # masks = [[False] * min(len(seq)+1, max_len) + [True] * (max_len - min(len(seq)+1, max_len)) for seq in data]
        masks = [[1] * min(len(seq), max_len) + [0] *
                (max_len - min(len(seq), max_len)) for seq in data]

        tensor_data = torch.tensor(padded_data, dtype=torch.float)
        tensor_mask = torch.tensor(masks, dtype=torch.long)
        return tensor_data, tensor_mask
    def _split_cn_sentence(self, sentence, use_sp=False):
        words = list(sentence)
        if use_sp:
            words = ["▁" if item == " " else item for item in words]
        return words
    def initialize_dataset(self, data_path, save_dir=''):
        processed_data_filename = Path(data_path).stem + "_processed.pkl"
        processed_data_path = os.path.join(save_dir, processed_data_filename)

        with open(data_path, 'r') as f:
            if data_path.endswith('json'):
                samples = json.load(f)
            else:
                samples = [json.loads(line) for line in f]

        samples_dict = {'features': [], 'prompt_len': [], 'label': [], 'text': []}

        for item in tqdm(samples):
            text = item['text']
            label = item['label']
            prompt_len = item['prompt_len']
        
            label_int = item['label_int']
            begin_idx_list = item['begin_idx_list']
            ll_tokens_list = item['ll_tokens_list']

            begin_idx_list = np.array(begin_idx_list)
            # Get the maximum value in begin_idx_list, which indicates where we need to truncate.
            max_begin_idx = np.max(begin_idx_list)
            # Truncate all vectors
            for idx, ll_tokens in enumerate(ll_tokens_list):
                ll_tokens_list[idx] = ll_tokens[max_begin_idx:]
            # Get the length of all vectors and take the minimum
            min_len = np.min([len(ll_tokens) for ll_tokens in ll_tokens_list])
            # Align the lengths of all vectors
            for idx, ll_tokens in enumerate(ll_tokens_list):
                ll_tokens_list[idx] = ll_tokens[:min_len]
            if len(ll_tokens_list) == 0 or len(ll_tokens_list[0]) == 0:
                continue
            ll_tokens_list = np.array(ll_tokens_list)
            # ll_tokens_list = normalize(ll_tokens_list, norm='l1')
            ll_tokens_list = ll_tokens_list.transpose()
            ll_tokens_list = ll_tokens_list.tolist()
            samples_dict['features'].append(ll_tokens_list)
            samples_dict['prompt_len'].append(prompt_len)
            samples_dict['label'].append(label)
            samples_dict['text'].append(text)
        
        # with open(processed_data_path, 'wb') as f:
        #     pickle.dump(samples_dict, f)

        return samples_dict
    def split_sentence(self, sentence, use_sp=False, cn_percent=0.2):
        total_char_count = len(sentence)
        total_char_count += 1 if total_char_count == 0 else 0
        chinese_char_count = sum('\u4e00' <= char <= '\u9fff' for char in sentence)
        if chinese_char_count / total_char_count > cn_percent:
            return self._split_cn_sentence(sentence, use_sp)
        else:
            return self._split_en_sentence(sentence, use_sp)
    def data_collator(self, samples):
            # samples: {'features': [], 'prompt_len': [], 'label': [], 'text': []}
            # batch: {'features': [], 'labels': [], 'text': []}
            batch = {}

            features = [sample['features'] for sample in samples]
            prompt_len = [sample['prompt_len'] for sample in samples]
            text = [sample['text'] for sample in samples]
            label = [sample['label'] for sample in samples]
            print("label-----"+str(label))
            features, masks = self.process_and_convert_to_tensor(features)
            # pad_masks = ~masks * -1
            pad_masks = (1 - masks) * self.label_pad_idx

            for idx, p_len in enumerate(prompt_len):
                prefix_len = len(self.split_sentence(text[idx][:p_len]))
                if prefix_len > self.max_len:
                    prefix_ids = self.sequence_labels_to_ids(self.max_len, self.human_label)
                    masks[idx][:] = prefix_ids[:]
                    continue
                total_len = len(self.split_sentence(text[idx]))
                
                if prefix_len > 0:
                    prefix_ids = self.sequence_labels_to_ids(prefix_len, self.human_label)
                    masks[idx][:prefix_len] = prefix_ids[:]
                if total_len - prefix_len > 0:
                    if total_len > self.max_len:
                        human_ids = self.sequence_labels_to_ids(self.max_len - prefix_len, label[idx])
                    else:
                        human_ids = self.sequence_labels_to_ids(total_len - prefix_len, label[idx])
                    masks[idx][prefix_len:total_len] = human_ids[:]
                masks[idx] += pad_masks[idx]

            batch['features'] = features
            #print("masks"+str(masks))
            batch['labels'] = masks
            batch['text'] = text
        # print(batch)
            return batch
    def get_eval_dataloader(self,dataset):
        return DataLoader(dataset,
                        batch_size=16,
                        sampler=SequentialSampler(dataset),
                        collate_fn=self.data_collator)
    def process_and_convert_to_tensor(self, data):
        """ here, data is features. """
        max_len = self.max_len
        # data shape: [B, S, E]

        feat_dim = len(data[0][0])
        padded_data = [  # [[0] * feat_dim] + 
            seq + [[0] * feat_dim] * (max_len - len(seq)) for seq in data
        ]
        padded_data = [seq[:max_len] for seq in padded_data]

        # masks = [[False] * min(len(seq)+1, max_len) + [True] * (max_len - min(len(seq)+1, max_len)) for seq in data]
        masks = [[1] * min(len(seq), max_len) + [0] *
                (max_len - min(len(seq), max_len)) for seq in data]

        tensor_data = torch.tensor(padded_data, dtype=torch.float)
        tensor_mask = torch.tensor(masks, dtype=torch.long)
        return tensor_data, tensor_mask
    def predict(self, test_path,model):
        data = dict()
        test_dict = self.initialize_dataset(test_path)
        data["test"] = Dataset.from_dict(test_dict)
        
        datasets = DatasetDict(data)
        if test_path:
            test_dataloader = self.get_eval_dataloader(datasets["test"])
        model.eval()
        model.to(self.device)
        texts = []
        true_labels = []
        embedings = []
        total_logits = []
        for step, inputs in enumerate(
                tqdm(test_dataloader, desc="Iteration")):
            try:
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                with torch.no_grad():
                    labels = inputs['labels']
                    #output = model(inputs['features'],inputs['text'], inputs['labels'])
                    output = model(inputs['features'], inputs['labels'])
                    tensor_cpu = output.cpu()
                    output = tensor_cpu.numpy()
                    embedings.extend(output)
                    true_labels.extend(labels.cpu().tolist())
                    #print(pred_labels)
                    # print(true_labels)
				
            except Exception as e:
                print(e)
                continue
        content_result = self.content_level_eval(true_labels)
        labels=np.array(content_result)
        embedings= np.vstack(embedings)
        print(embedings)
        # 创建UMAP对象
        reducer = umap.UMAP()
        # 拟合数据并降维
        embedding = reducer.fit_transform(embedings)
        print(embedding)
        # 使用Matplotlib绘制降维后的数据，并为不同的标签设置不同的颜色
        unique_labels = np.unique(labels)
        print(unique_labels)
        u_labels=[]
        for label in unique_labels:
            if label=="gpt3":
                u_labels.append('GPT-3.5')
                
            elif label=="gpt4":
                u_labels.append('GPT-4o')
            elif label=="gpt4A":
                u_labels.append('GPT-4o-M')
            elif label=="gpt3A":
                u_labels.append('GPT-3.5-M')
            elif label=="qwen":
                u_labels.append('Qwen2')
            elif label=="llama":
                u_labels.append('LLaMA')
            elif label=="human":
                u_labels.append('Human')
            elif label=="gpt2":
                u_labels.append('GPT-2')
            else:
                u_labels.append(label)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        plt.figure(figsize=(10, 8))
        for i, label in enumerate(unique_labels):
                plt.scatter(embedding[labels == label, 0], embedding[labels == label, 1], 
                                        c=colors[i], label=label, s=10)

        plt.legend(u_labels,fontsize=12)
        plt.grid(True, which='both', linewidth=0.5)
        # 设置主要刻度定位器
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(5))

        # 设置次要刻度定位器
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))
        # plt.yticks([])
        plt.xticks([])
        plt.yticks([])
        plt.title('UMAP Embedding with Different Colors for Labels')
        # plt.show()
        plt.axis('equal')
        plt.savefig('umap_embedding.png') 
        #print(content_result)
    def _get_most_common_tag(self, tags):
        """most_common_tag is a tuple: (tag, times)"""
        from collections import Counter

        tags = [self.id2label[tag] for tag in tags]
        tags = [tag.split('-')[-1] for tag in tags]
        tag_counts = Counter(tags)
        most_common_tag = tag_counts.most_common(1)[0]

        return most_common_tag	
    def content_level_eval(self,  true_labels):
        #from collections import Counter

        true_content_labels = []
        
        for true_label in true_labels:
            true_label = np.array(true_label)
            mask = true_label != -1
            true_label = true_label[mask].tolist()
            true_common_tag = self._get_most_common_tag(true_label)
            true_content_labels.append(true_common_tag[0])
        
        return true_content_labels
if __name__=="__main__":
    test_path='/mnt/chenxi/project/SeqXGPT/SeqXGPT/dataset/data_new/testk.jsonl'
    dm=DataManager()
    id2label,label2id,_=dm.get_label2id()
    model = ModelWiseTransformerClassifier(id2labels=id2label, seq_len=512)
    saved_model = torch.load(ckpt_path)
    model.load_state_dict(saved_model.state_dict(),strict=False)
    model.eval()
    dm.predict(test_path,model)
