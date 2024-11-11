import torch
import torch.nn as nn
import nltk
from typing import List, Tuple
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers.models.bert import BertModel
from fastNLP.modules.torch import MLP,ConditionalRandomField,allowed_transitions
from torch.nn import CrossEntropyLoss
from transformers import RobertaTokenizer, RobertaModel
from transformers import T5Tokenizer, T5EncoderModel
from nltk import pos_tag
from nltk.tokenize import word_tokenize
class ConvFeatureExtractionModel(nn.Module):

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        conv_dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride=1, conv_bias=False):
            padding = k // 2
            return nn.Sequential(
                nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=k, stride=stride, padding=padding, bias=conv_bias),
                nn.Dropout(conv_dropout),
                # nn.BatchNorm1d(n_out),
                nn.ReLU(),
                # nn.MaxPool1d(kernel_size=2, stride=2)
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for _, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(in_d, dim, k, stride=stride, conv_bias=conv_bias))
            in_d = dim

    def forward(self, x):
        # x = x.unsqueeze(1)
       # print(self.conv_layers)
        for conv in self.conv_layers:
            x = conv(x)
        return x


class ModelWiseCNNClassifier(nn.Module):

    def __init__(self, id2labels, dropout_rate=0.1):
        super(ModelWiseCNNClassifier, self).__init__()
        feature_enc_layers = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]
        self.conv = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            conv_dropout=0.0,
            conv_bias=False,
        )

        embedding_size = 4 *64
        self.norm = nn.LayerNorm(embedding_size)
        
        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(embedding_size, self.label_num))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=allowed_transitions(id2labels))
        self.crf.trans_m.data *= 0

    def conv_feat_extract(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out

    def forward(self, x, labels):
        x = x.transpose(1, 2)
        out1 = self.conv_feat_extract(x[:, 0:1, :])  
        out2 = self.conv_feat_extract(x[:, 1:2, :])  
        out3 = self.conv_feat_extract(x[:, 2:3, :])  
        out4 = self.conv_feat_extract(x[:, 3:4, :])  
        outputs = torch.cat((out1, out2, out3, out4), dim=2)  
        
        outputs = self.norm(outputs)
        dropout_outputs = self.dropout(outputs)
        logits = self.classifier(dropout_outputs)
        
        if self.training:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.label_num), labels.view(-1))
            output = {'loss': loss, 'logits': logits}
        else:
            mask = labels.gt(-1)
            paths, scores = self.crf.viterbi_decode(logits=logits, mask=mask)
            paths[mask==0] = -1
            output = {'preds': paths, 'logits': logits}
            pass

        return output
    

class ModelWiseTransformerClassifier(nn.Module):

    def __init__(self, id2labels, seq_len, intermediate_size = 512, num_layers=2, dropout_rate=0.1):
        super(ModelWiseTransformerClassifier, self).__init__()
        # feature_enc_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]
        feature_enc_layers = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]
        self.conv = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            conv_dropout=0.0,
            conv_bias=False,
        )
        
        self.seq_len = seq_len          # MAX Seq_len
        embedding_size = 5*64
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=16,
            dim_feedforward=intermediate_size,
            dropout=dropout_rate,
            batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                            num_layers=num_layers)

        self.position_encoding = torch.zeros((seq_len, embedding_size))
        for pos in range(seq_len):
            for i in range(0, embedding_size, 2):
                self.position_encoding[pos, i] = torch.sin(
                    torch.tensor(pos / (10000**((2 * i) / embedding_size))))
                self.position_encoding[pos, i + 1] = torch.cos(
                    torch.tensor(pos / (10000**((2 *
                                                 (i + 1)) / embedding_size))))
        
        self.norm = nn.LayerNorm(embedding_size)
        self.layer_norm = nn.LayerNorm(768)        
        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(embedding_size+768*3, self.label_num))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=allowed_transitions(id2labels))
        self.crf.trans_m.data *= 0
        #self.tokenizer = T5Tokenizer.from_pretrained('/mnt/chenxi/project/SeqXGPT/SeqXGPT/model/T5')
        #self.model=T5EncoderModel.from_pretrained('/mnt/chenxi/project/SeqXGPT/SeqXGPT/model/T5')
        self.tokenizer = RobertaTokenizer.from_pretrained('/mnt/chenxi/project/SeqXGPT/SeqXGPT/model/roberta')
        self.model = RobertaModel.from_pretrained('/mnt/chenxi/project/SeqXGPT/SeqXGPT/model/roberta')

        
    def conv_feat_extract(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out
		
    def get_output(self,text):
        encoded_input = self.tokenizer(text, return_tensors='pt',max_length=self.seq_len, truncation=True, pad_to_max_length=True)
        self.model.to('cuda')
        encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}

        input_ids = encoded_input['input_ids'][:self.seq_len]
        attention_mask = encoded_input['attention_mask'][:self.seq_len]
        # 调用模型，获取输出
        output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return output
    def get_output_bigram(self,text):
        encoded_input = self.tokenizer(text, return_tensors='pt',max_length=self.seq_len, truncation=True, pad_to_max_length=True)
        self.model.to('cuda')
        encoded_input = {k: v.to('cuda') for k, v in encoded_input.items()}

        input_ids = encoded_input['input_ids'][:self.seq_len]
        attention_mask = encoded_input['attention_mask'][:self.seq_len]
        # 调用模型，获取输出
        output = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        bigram_features = []
        word_embeddings=output
        # 提取bigram特征
          # 提取bigram特征
        sentence_bigram_features = []
        for i in range(input_ids.size(0)):
            # 获取当前句子的词嵌入
            word_embeddings = output[i]
            bigram_features_for_sentence = []
            for j in range(len(input_ids[i]) - 1):
                # 获取当前bigram的两个词的嵌入
                word1_embedding = word_embeddings[j, :]
                word2_embedding = word_embeddings[j + 1, :]
                # 计算两个词嵌入的平均值
                bigram_embedding = (word1_embedding + word2_embedding) / 2
                bigram_features_for_sentence.append(bigram_embedding.unsqueeze(0))
            # 检查bigram特征的数量
            num_bigrams = len(bigram_features_for_sentence)
            # 如果bigram特征的数量少于512，使用零填充
            if num_bigrams < 512:
                padding_size = 512 - num_bigrams
                padding_tensor = torch.zeros((padding_size, word_embeddings.size(1))).to(word_embeddings.device)
                bigram_features_for_sentence.extend([padding_tensor] * padding_size)
            sentence_bigram_features.append(torch.cat(bigram_features_for_sentence, dim=0))

        # 将每个句子的bigram特征拼接起来形成一个批量的bigram特征张量
        batch_bigram_features_tensor = torch.stack(sentence_bigram_features, dim=0)
        # 打印bigram特征向量的形状
        #print(batch_bigram_features_tensor.shape)
        return batch_bigram_features_tensor,output
        # 将每个句子的bigram特征拼接起来
    def forward(self, x, text,labels):
        mask = labels.gt(-1)
        pos=[]
        for t in text:
            words=pos_tag(word_tokenize(t))
            pos.append(" ".join([word[1] for word in words]))
        pos_output=self.get_output(pos)
        bigram_features,text_output=self.get_output_bigram(text)	
        pos_output=self.layer_norm(pos_output)
        text_output=self.layer_norm(text_output)
        bigram_features=self.layer_norm(text_output)
        pb_outputs=torch.cat((pos_output,text_output,bigram_features),dim=2)
        padding_mask = ~mask
        x = x.transpose(1, 2)
        out1 = self.conv_feat_extract(x[:, 0:1, :])  
        out2 = self.conv_feat_extract(x[:, 1:2, :])  
        out3 = self.conv_feat_extract(x[:, 2:3, :])  
        out4 = self.conv_feat_extract(x[:, 3:4, :])
        out5 = self.conv_feat_extract(x[:, 4:5, :])
        out = torch.cat((out1, out2, out3, out4,out5), dim=2)  
        #out=torch.cat((out1, out4,out5), dim=2)
        outputs = out + self.position_encoding.to(out.device)
        outputs = self.norm(outputs)
        #print("norm.shape-------"+str(outputs.shape))
        outputs = self.encoder(outputs, src_key_padding_mask=padding_mask)
        outputs=torch.cat((outputs,pb_outputs),dim=2)
        output = outputs.mean(dim=1)
        dropout_outputs = self.dropout(outputs)
        logits = self.classifier(dropout_outputs)
        if self.training:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.label_num), labels.view(-1))
            output = {'loss': loss, 'logits': logits}
        else:
            paths, scores = self.crf.viterbi_decode(logits=logits, mask=mask)
            paths[mask==0] = -1
            output = {'preds': paths, 'logits': logits}
            pass
        return output
    

class TransformerOnlyClassifier(nn.Module):

    def __init__(self, id2labels, seq_len, embedding_size=4, num_heads=2, intermediate_size=64, num_layers=2, dropout_rate=0.1):
        super(TransformerOnlyClassifier, self).__init__()

        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout_rate,
            batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                            num_layers=num_layers)

        self.position_encoding = torch.zeros((seq_len, embedding_size))
        for pos in range(seq_len):
            for i in range(0, embedding_size, 2):
                self.position_encoding[pos, i] = torch.sin(
                    torch.tensor(pos / (10000**((2 * i) / embedding_size))))
                self.position_encoding[pos, i + 1] = torch.cos(
                    torch.tensor(pos / (10000**((2 *
                                                 (i + 1)) / embedding_size))))
        
        self.norm = nn.LayerNorm(embedding_size)
        
        self.label_num = len(id2labels)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(nn.Linear(embedding_size, self.label_num))
        self.crf = ConditionalRandomField(num_tags=self.label_num, allowed_transitions=allowed_transitions(id2labels))
        self.crf.trans_m.data *= 0
    
    def forward(self, inputs, labels):
        mask = labels.gt(-1)
        padding_mask = ~mask
        
        outputs = inputs + self.position_encoding.to(inputs.device)
        outputs = self.norm(outputs)
        outputs = self.encoder(outputs, src_key_padding_mask=padding_mask)
        dropout_outputs = self.dropout(outputs)
        logits = self.classifier(dropout_outputs)
        
        if self.training:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.label_num), labels.view(-1))
            output = {'loss': loss, 'logits': logits}
        else:
            paths, scores = self.crf.viterbi_decode(logits=logits, mask=mask)
            paths[mask==0] = -1
            output = {'preds': paths, 'logits': logits}
            pass

        return output
