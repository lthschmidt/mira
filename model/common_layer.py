### MOST OF IT TAKEN FROM https://github.com/kolloldas/torchnlp
import matplotlib
# matplotlib.use('Agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import numpy as np
import math
import os
import io #new
from utils import config
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from utils.metric import rouge, moses_multi_bleu, _prec_recall_f1_score, compute_prf, compute_exact_match
from transformers import AutoTokenizer, AutoModel

# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Импорт Translator в зависимости от модели
if config.model == 'trs':
    from utils.beam_omt import Translator
elif config.model == 'seq2seq':
    from utils.beam_ptr import Translator
elif config.model == 'multi-trs':
    from utils.beam_omt_multiplex import Translator
elif config.model == 'experts':
    from utils.beam_omt_experts import Translator
elif config.model == 'mimic':
    from utils.beam_omt_mimic_model1 import Translator
elif config.model == 'vae':
    from utils.beam_omt_vae import Translator
elif config.model == 'new_dec':
    from utils.beam_omt_mimic_new_dec import Translator
elif config.model == 'new_dec2':
    from utils.beam_omt_mimic_new_dec2 import Translator
elif config.model == 'model1_bert':
    from utils.beam_omt_bert import Translator
elif config.model == 'model1_noMimic':
    from utils.beam_omt_noMimic import Translator
elif config.model == 'model1_gpt2':
    from utils.beam_omt_gpt2 import Translator

import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import numpy as np

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                       hidden_size, num_heads, bias_mask, attention_dropout)
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding='both',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, mask=None):
        x = inputs
        x_norm = self.layer_norm_mha(x)
        y, _ = self.multi_head_attention(x_norm, x_norm, x_norm, mask)
        x = self.dropout(x + y)
        x_norm = self.layer_norm_ffn(x)
        y = self.positionwise_feed_forward(x_norm)
        y = self.dropout(x + y)
        return y

class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                       hidden_size, num_heads, bias_mask, attention_dropout)
        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                       hidden_size, num_heads, None, attention_dropout)
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='cc', padding='left',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x, encoder_outputs, attention_weight, mask = inputs
        mask_src, dec_mask = mask
        x_norm = self.layer_norm_mha_dec(x)
        y, _ = self.multi_head_attention_dec(x_norm, x_norm, x_norm, dec_mask)
        x = self.dropout(x + y)
        x_norm = self.layer_norm_mha_enc(x)
        y, attention_weight = self.multi_head_attention_enc_dec(x_norm, encoder_outputs, encoder_outputs, mask_src)
        x = self.dropout(x + y)
        x_norm = self.layer_norm_ffn(x)
        y = self.positionwise_feed_forward(x_norm)
        y = self.dropout(x + y)
        return y, encoder_outputs, attention_weight, mask

class MultiExpertMultiHeadAttention(nn.Module):
    def __init__(self, num_experts, input_depth, total_key_depth, total_value_depth, output_depth,
                 num_heads, bias_mask=None, dropout=0.0):
        """
        Parameters:
            expert_num: Number of experts
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiExpertMultiHeadAttention, self).__init__()
        if total_key_depth % num_heads != 0:
            print("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))
            total_value_depth = total_value_depth - (total_value_depth % num_heads)
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.query_scale = (total_key_depth//num_heads)**-0.5 ## sqrt
        self.bias_mask = bias_mask
        self.query_linear = nn.Linear(input_depth, total_key_depth*num_experts, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth*num_experts, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth*num_experts, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth*num_experts, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_experts, self.num_heads, shape[2]//(self.num_heads*self.num_experts)).permute(0, 2, 3, 1, 4)

    def _merge_heads(self, x):
        if len(x.shape) != 5:
            raise ValueError("x must have rank 5")
        shape = x.shape
        return x.permute(0, 3, 1, 2, 4).contiguous().view(shape[0], shape[3], self.num_experts, shape[4]*self.num_heads)

    def forward(self, queries, keys, values, mask):
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        queries *= self.query_scale
        logits = torch.matmul(queries, keys.permute(0, 1, 2, 4, 3))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 1, T_values]
            logits = logits.masked_fill(mask, -1e18)
        weights = nn.functional.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        contexts = torch.matmul(weights, values)
        contexts = self._merge_heads(contexts)
        outputs = self.output_linear(contexts)

        return outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth,
                 num_heads, bias_mask=None, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        if total_key_depth % num_heads != 0:
            print("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
            total_key_depth = total_key_depth - (total_key_depth % num_heads)
        if total_value_depth % num_heads != 0:
            print("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))
            total_value_depth = total_value_depth - (total_value_depth % num_heads)
        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        self.bias_mask = bias_mask
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values, mask):
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        queries *= self.query_scale
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        if mask is not None:
            mask = mask.unsqueeze(1)
            logits = logits.masked_fill(mask, -1e18)
        weights = nn.functional.softmax(logits, dim=-1)
        weights = self.dropout(weights)
        contexts = torch.matmul(weights, values)
        contexts = self._merge_heads(contexts)
        outputs = self.output_linear(contexts)
        return outputs, logits.sum(dim=1) / self.num_heads

class Conv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, pad_type):
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)
        return outputs

class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)] * (len(layer_config) - 2) +
                 [(filter_size, output_depth)])
        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def _gen_bias_mask(max_length):
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.tensor(np_mask, dtype=torch.float32)
    return torch_mask.unsqueeze(0).unsqueeze(1)

def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float64) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])
    return torch.tensor(signal, dtype=torch.float32)

def _get_attn_subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.tensor(subsequent_mask)
    return subsequent_mask.to(device)

class OutputLayer(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(OutputLayer, self).__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)

    def loss(self, hidden, labels):
        raise NotImplementedError('Must implement {}.loss'.format(self.__class__.__name__))

class SoftmaxOutputLayer(OutputLayer):
    def forward(self, hidden):
        logits = self.output_projection(hidden)
        probs = F.softmax(logits, -1)
        _, predictions = torch.max(probs, dim=-1)
        return predictions

    def loss(self, hidden, labels):
        logits = self.output_projection(hidden)
        log_probs = F.log_softmax(logits, -1)
        return F.nll_loss(log_probs.view(-1, self.output_size), labels.view(-1))

def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_dim + 1) / 2) * (j - (sentence_size + 1) / 2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    return np.transpose(encoding)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        logp = F.log_softmax(input, dim=-1)
        ce_loss = F.nll_loss(logp, target, reduction='none')  # обычная cross-entropy
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def gen_embeddings(vocab):
    embeddings = np.random.randn(vocab.n_words, config.emb_dim) * 0.01
    print(f'Embeddings: {vocab.n_words} x {config.emb_dim}')
    
    if not config.emb_file:
        return embeddings

    pre_trained = 0
    with io.open(config.emb_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        n_words_file, emb_dim_file = map(int, f.readline().split())  # Читаем заголовок
        if emb_dim_file != config.emb_dim:
            raise ValueError(f"Embedding dim mismatch: file={emb_dim_file}, config={config.emb_dim}")

        for line in f:
            parts = line.rstrip().split(' ')
            word = parts[0]
            if word in vocab.word2index:
                try:
                    vector = np.array(parts[1:], dtype=np.float32)
                    embeddings[vocab.word2index[word]] = vector
                    pre_trained += 1
                except (ValueError, IndexError) as e:
                    print(f"Skipping {word}: {str(e)}")
    
    print(f'Pre-trained: {pre_trained} ({pre_trained/vocab.n_words:.1%})')
    return embeddings

# def gen_bert_embeddings(vocab):
#     tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
#     model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
#     model.eval()  # Важно — режим инференса

#     # emb_matrix = model.get_input_embeddings().weight.data  # [vocab_size, hidden_size]
#     # emb_dim = emb_matrix.shape[1]

#     # print(f'Embeddings (BERT-based): {emb_matrix.shape[0]} x {emb_dim}')
#     # print(f'Pre-trained: {emb_matrix.shape[0]} (100.0%)')
#     # return emb_matrix.cpu().numpy()
#     with torch.no_grad():
#         # Получаем эмбеддинги для BERT-части (веса токенов, без спецтокенов)
#         bert_embeddings = model.get_input_embeddings().weight.cpu().numpy()
    
#     bert_vocab_size, emb_dim = bert_embeddings.shape
#     vocab_size = vocab.n_words
#     special_offset = vocab_size - bert_vocab_size  # напр., 83835 - 83828 = 7

#     # Проверка на соответствие размеров
#     assert special_offset >= 0, "Что-то не так с размером словаря!"

#     # Создаём полную матрицу эмбеддингов (случайная инициализация спецтокенов)
#     full_embeddings = np.random.normal(scale=0.02, size=(vocab_size, emb_dim)).astype(np.float32)

#     # Вставляем BERT-эмбеддинги со сдвигом
#     full_embeddings[special_offset:] = bert_embeddings
#     return full_embeddings

class Embeddings(nn.Module):
    def __init__(self, vocab, d_model, padding_idx=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

# def share_embedding(vocab, pretrain=True):
#     # tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
#     # vocab_size = tokenizer.vocab_size
#     embedding = Embeddings(vocab.n_words, config.emb_dim, padding_idx=config.PAD_idx)
#     if pretrain:
#         #pre_embedding = gen_embeddings(vocab)
#         pre_embedding = gen_bert_embeddings(vocab)
#         assert pre_embedding.shape == (vocab.n_words, config.emb_dim), \
#             f"Shape mismatch: got {pre_embedding.shape}, expected {(vocab.n_words, config.emb_dim)}"
#         embedding.lut.weight.data.copy_(torch.tensor(pre_embedding, dtype=torch.float32))
#         embedding.lut.weight.data.requires_grad = True
#     return embedding

def share_embedding(vocab, pretrain=True):
    embedding = Embeddings(vocab.n_words, config.emb_dim, padding_idx=config.PAD_idx)
    if pretrain:
        pre_embedding = gen_embeddings(vocab)
        embedding.lut.weight.data.copy_(torch.tensor(pre_embedding, dtype=torch.float32))
        embedding.lut.weight.data.requires_grad = True
    return embedding

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.size()[0] > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        return self.optimizer.state_dict()

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(config.PAD_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_input_from_batch(batch):
    enc_batch = batch["input_batch"]
    enc_lens = batch["input_lengths"]
    batch_size, max_enc_len = enc_batch.size()
    assert len(enc_lens) == batch_size
    enc_lens=enc_lens.to(config.device)
    enc_padding_mask = sequence_mask(enc_lens, max_len=max_enc_len).float()

    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = batch["input_ext_vocab_batch"]
        if batch["max_art_oovs"] > 0:
            extra_zeros = torch.zeros((batch_size, batch["max_art_oovs"]))

    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim))

    coverage = None
    if config.is_coverage:
        coverage = torch.zeros(enc_batch.size())

    # Переносим все на устройство
    enc_padding_mask = enc_padding_mask.to(device)
    if enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab = enc_batch_extend_vocab.to(device)
    if extra_zeros is not None:
        extra_zeros = extra_zeros.to(device)
    c_t_1 = c_t_1.to(device)
    if coverage is not None:
        coverage = coverage.to(device)

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage

def get_output_from_batch(batch):
    dec_batch = batch["target_batch"]

    if config.pointer_gen:
        target_batch = batch["target_ext_vocab_batch"]
    else:
        target_batch = dec_batch

    dec_lens_var = batch["target_lengths"]
    max_dec_len = max(dec_lens_var)

    assert max_dec_len == target_batch.size(1)

    dec_padding_mask = sequence_mask(dec_lens_var, max_len=max_dec_len).float()

    return dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long().to(sequence_length.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = (sequence_length.unsqueeze(1)
                        .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def write_config():
    if not config.test:
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path)
        with open(os.path.join(config.save_path, 'config.txt'), 'w') as the_file:
            for k, v in config.args.__dict__.items():
                if "False" in str(v):
                    pass
                elif "True" in str(v):
                    the_file.write(f"--{k} ")
                else:
                    the_file.write(f"--{k} {v} ")

def print_custum(emotion, dial, ref, hyp_g, hyp_b, hyp_t):
    print(f"emotion: {emotion}")
    print(f"Context: {dial}")
    print(f"Topk: {hyp_t}")
    print(f"Beam: {hyp_b}")
    print(f"Greedy: {hyp_g}")
    print(f"Ref: {ref}")
    print("----------------------------------------------------------------------")
    print("----------------------------------------------------------------------")

def write_custum(emotion, vader_score, emo, dial, ref, hyp_g, hyp_b, hyp_t):
    ret = f"emotion: {emotion}\t" + f'vader-score: {vader_score}\t' + f'predicted_emotion: {emo}\n'
    ret += f"Context: {dial}\n"
    ret += f"Topk: {hyp_t}\n"
    ret += f"Beam: {hyp_b}\n"
    ret += f"Greedy: {hyp_g}\n"
    ret += f"Ref: {ref}\n"
    ret += "----------------------------------------------------------------------\n"
    return ret

def evaluate(model, data, ty='valid', max_dec_step=30, write_summary=False):
    emo_map = {
        'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
        'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13, 'anxious': 14, 'disappointed': 15,
        'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21, 'anticipating': 22, 'embarrassed': 23,
        'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29, 'apprehensive': 30, 'faithful': 31}
    emo_map = {v: k for k, v in emo_map.items()}

    model.__id__logger = 0
    dial = []
    ref, hyp_g, hyp_b, hyp_t = [], [], [], []
    if ty == "test":
        print("testing generation:")
    t = Translator(model, model.vocab)
    l = []
    p = []
    bce = []
    acc = []
    pbar = tqdm(enumerate(data), total=len(data))
    inf_results = []
    smoother = SmoothingFunction()
    try:
        for j, batch in pbar:
            loss, ppl, bce_prog, acc_prog = model.train_one_batch(batch, 0, train=False)
            l.append(loss)
            p.append(ppl)
            bce.append(bce_prog)
            acc.append(acc_prog)
            if ty == "test":
                sent_g, vader_score, emotion_id = model.decoder_greedy(batch, max_dec_step=max_dec_step)
                sent_b = t.beam_search(batch, max_dec_step=max_dec_step)
                sent_t = model.decoder_topk(batch, max_dec_step=max_dec_step)
                for i, (greedy_sent, beam_sent, topk_sent) in enumerate(zip(sent_g, sent_b, sent_t)):
                    rf = " ".join(batch["target_txt"][i])
                    hyp_g.append(greedy_sent)
                    hyp_b.append(beam_sent)
                    hyp_t.append(topk_sent)
                    ref.append(rf)
                    temp = write_custum(emotion=batch["program_txt"][i], vader_score=vader_score, emo=emo_map[emotion_id],
                                       dial=[" ".join(s) for s in batch['input_txt'][i]] if config.dataset == "empathetic" else " ".join(batch['input_txt'][i]),
                                       ref=rf, hyp_t=topk_sent, hyp_g=greedy_sent, hyp_b=beam_sent)
                    inf_results.append(temp)
                #pbar.set_description(f"loss: {np.mean(l):.4f} ppl: {math.exp(np.mean(l)):.1f}")
            #pass
    except KeyboardInterrupt:
        print("Only testing for a fraction of testing dataset, do not use this result!")

    loss = np.mean(l)
    ppl = np.mean(p)
    bce = np.mean(bce)
    acc = np.mean(acc)
    # Заменяем moses_multi_bleu на NLTK реализацию
    def calc_bleu(hypotheses, references):
        # Подготовка данных для NLTK BLEU
        refs = [[ref.split()] for ref in references]
        hyps = [hyp.split() for hyp in hypotheses]
        try:
            return corpus_bleu(refs, hyps, smoothing_function=smoother.method1) * 100
        except:
            return 0.0  # Возвращаем 0 если не удалось вычислить
    
    bleu_score_g = calc_bleu(hyp_g, ref)
    bleu_score_b = calc_bleu(hyp_b, ref)
    bleu_score_t = calc_bleu(hyp_t, ref)

    print("\nEVAL\tLoss\tPPL\tAccuracy\tBleu_g\tBleu_b\tBlue_t")
    print(f"{ty}\t{loss:.4f}\t{math.exp(loss):.4f}\t{acc:.2f}\t{bleu_score_g:.2f}\t{bleu_score_b:.2f}\t{bleu_score_t:.2f}")
    if write_summary:
        return loss, math.exp(loss), bce, acc, bleu_score_g, bleu_score_b, bleu_score_t, inf_results
    else:
        return loss, math.exp(loss), bce, acc, bleu_score_g, bleu_score_b, bleu_score_t

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x

def top_k_top_p_filtering(logits, top_k=0, top_p=0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits