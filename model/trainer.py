import sys, os, time, math, random
from copy import deepcopy
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter

from utils.data_loader import prepare_data_seq
from utils import config
from model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm, \
    _gen_bias_mask, _gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask, \
    get_input_from_batch, get_output_from_batch, top_k_top_p_filtering, FocalLoss
from model.transformer_mulexpert import Encoder, Decoder, MulDecoder, Generator, MulDecoder, ACT_basic
from model.emotion_input_attention import EmotionInputEncoder
from model.complex_res_attention import ComplexResDecoder
from model.complex_res_gate import ComplexResGate
from model.decoder_context_v import DecoderContextV
from model.VAE_noEmo_posterior import VAESampling
#from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def freeze_layers(module, n_unfrozen=0):
    # Перебираем все слои
    for i, child in enumerate(module.children()):
        # Замораживаем или размораживаем слой
        for param in child.parameters():
            param.requires_grad = i >= n_unfrozen

class Train_MIME(nn.Module):
    '''
    Model for training with emotion attention.
    Emotion is passed as the Q in a decoder block of the transformer.
    '''

    def __init__(self, vocab, decoder_number, model_file_path=None, is_eval=False, load_optim=False):
        super(Train_MIME, self).__init__()
        # self.embedding = share_embedding(config.pretrain_emb)
        # tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
        # self.vocab_size = tokenizer.vocab_size
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.embedding = share_embedding(self.vocab, config.pretrain_emb)

        self.encoder = Encoder(
            config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
            total_key_depth=config.depth, total_value_depth=config.depth,
            filter_size=config.filter, universal=config.universal, input_dropout=config.input_dropout,
            layer_dropout=config.layer_dropout,
            attention_dropout=config.attention_dropout,
            relu_dropout=config.relu_dropout
        )
        self.decoder_number = decoder_number

        self.decoder = DecoderContextV(
            config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
            total_key_depth=config.depth, total_value_depth=config.depth, filter_size=config.filter
        )

        self.vae_sampler = VAESampling(config.hidden_dim, config.hidden_dim, out_dim=312)

        # Outputs m
        self.emotion_input_encoder_1 = EmotionInputEncoder(
            config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
            total_key_depth=config.depth, total_value_depth=config.depth,
            filter_size=config.filter, universal=config.universal, emo_input_mode=config.emo_input
        )
        # Outputs m~
        self.emotion_input_encoder_2 = EmotionInputEncoder(
            config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
            total_key_depth=config.depth, total_value_depth=config.depth,
            filter_size=config.filter, universal=config.universal, emo_input_mode=config.emo_input
        )

        if config.emo_combine == "att":
            self.cdecoder = ComplexResDecoder(
                config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                total_key_depth=config.depth, total_value_depth=config.depth, filter_size=config.filter,
                input_dropout=config.input_dropout, layer_dropout=config.layer_dropout, attention_dropout=config.attention_dropout,
                relu_dropout=config.relu_dropout
            )
        elif config.emo_combine == "gate":
            self.cdecoder = ComplexResGate(config.emb_dim)

        self.s_weight = nn.Linear(config.hidden_dim, config.emb_dim, bias=False)
        self.decoder_key = nn.Linear(config.hidden_dim, decoder_number, bias=False)

        # v^T tanh(W E[i] + H c + b)
        if True:  # Method 3
            self.e_weight = nn.Linear(config.emb_dim, config.emb_dim, bias=True)
            self.v = torch.rand(config.emb_dim, requires_grad=True).to(device)

        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.emoji_embedding = nn.Embedding(32, config.emb_dim)
        if config.init_emo_emb:
            self.init_emoji_embedding_with_glove()

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion1 = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=config.PAD_idx)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        if config.softmax:
            self.attention_activation = nn.Softmax(dim=1)
        else:
            self.attention_activation = nn.Sigmoid()  # nn.Softmax()

        #self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(
                config.hidden_dim, 1, 8000,
                torch.optim.Adam(self.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
            )
        else:            
            # оптимизатор
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=config.lr, weight_decay=0.01)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',  # уменьшаем lr, если метрика (например, perplexity) не уменьшается
                factor=0.8,  # во сколько раз уменьшить lr
                patience=2  # сколько эпох ждать перед уменьшением
            )

        # VAE
        self.kl_annealing_steps = 10000  # за сколько шагов дойдёт до полного веса
        self.kl_max_weight = 1.0  # максимальный вес KL (обычно 1.0)
        # Новый код с адаптацией BERT эмбеддингов
        self.bert_adapter = nn.Linear(config.emb_dim, config.emb_dim).to(device)


        if model_file_path is not None:
            print("Loading weights...")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.load_state_dict(state['model'])
            if load_optim:
                self.optimizer.load_state_dict(state['optimizer'])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

        # Positive and negative emotions
        self.positive_emotions = [11, 16, 6, 8, 3, 1, 28, 13, 31, 17, 24, 0, 27]
        self.negative_emotions = [9, 4, 2, 22, 14, 30, 29, 25, 15, 10, 23, 19, 18, 21, 7, 20, 5, 26, 12]


    def init_emoji_embedding_with_glove(self):
        emotions = [
            'surprised', 'excited', 'annoyed', 'proud', 'angry', 'sad', 'grateful', 'lonely',
            'impressed', 'afraid', 'disgusted', 'confident', 'terrified', 'hopeful',
            'anxious', 'disappointed', 'joyful', 'prepared', 'guilty', 'furious', 'nostalgic', 'jealous',
            'anticipating', 'embarrassed', 'content', 'devastated', 'sentimental', 'caring', 'trusting',
            'ashamed', 'apprehensive', 'faithful'
        ]
        emotion_index = [self.vocab.word2index[i] for i in emotions]
        emoji_embedding_init = self.embedding(torch.tensor(emotion_index, dtype=torch.long))
        self.emoji_embedding.weight.data = emoji_embedding_init
        self.emoji_embedding.weight.requires_grad = True

    def get_kl_weight(self, step):
        return min(self.kl_max_weight, step / self.kl_annealing_steps)

    def save_model(self, running_avg_ppl, iter, f1_g, f1_b, ent_g, ent_b, ent_t):
        state = {
            'iter': iter,
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl,
            'model': self.state_dict()
        }
        model_save_path = os.path.join(
            self.model_dir,
            f'model_{iter}_{running_avg_ppl:.4f}_{f1_g:.4f}_{f1_b:.4f}_{ent_g:.4f}_{ent_b:.4f}_{ent_t:.4f}'
        )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def random_sampling(self, e):
        p = np.random.choice(self.positive_emotions)
        n = np.random.choice(self.negative_emotions)
        if e in self.positive_emotions:
            mimic = p
            mimic_t = n
        else:
            mimic = n
            mimic_t = p
        return mimic, mimic_t

    def train_one_batch(self, batch, iter, train=True):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()

        # Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)

        if config.dataset == "empathetic":
            emb_mask = self.embedding(batch["mask_input"])
            encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)
        else:
            encoder_outputs = self.encoder(self.embedding(enc_batch), mask_src)

        #q_h= torch.mean(encoder_outputs, dim=1) if config.mean_query else encoder_outputs[:, 0]
        # q_h, _ = torch.max(encoder_outputs, dim=1)
        q_h_max, _ = torch.max(encoder_outputs, dim=1)
        q_h_mean = torch.mean(encoder_outputs, dim=1)
        q_h = 0.6 * q_h_max + 0.4 * q_h_mean
        x = self.s_weight(q_h)
        logit_prob = torch.matmul(x, self.emoji_embedding.weight.transpose(0,1))
        # VAE Sampling
        emotions_mimic, emotions_non_mimic, mu_positive_prior, logvar_positive_prior, mu_negative_prior, logvar_negative_prior = \
            self.vae_sampler(q_h, batch['program_label'], self.emoji_embedding)

        m_out = self.emotion_input_encoder_1(emotions_mimic.unsqueeze(1), encoder_outputs, mask_src)
        m_tilde_out = self.emotion_input_encoder_2(emotions_non_mimic.unsqueeze(1), encoder_outputs, mask_src)
        if train:
            emotions_mimic, emotions_non_mimic, mu_positive_posterior, logvar_positive_posterior, mu_negative_posterior, logvar_negative_posterior = \
                self.vae_sampler.forward_train(q_h, batch['program_label'], self.emoji_embedding, M_out=m_out.mean(dim=1), M_tilde_out=m_tilde_out.mean(dim=1))
            KLLoss_positive = self.vae_sampler.kl_div(mu_positive_posterior, logvar_positive_posterior, mu_positive_prior, logvar_positive_prior)
            KLLoss_negative = self.vae_sampler.kl_div(mu_negative_posterior, logvar_negative_posterior, mu_negative_prior, logvar_negative_prior)
            KLLoss = KLLoss_positive + KLLoss_negative
        else:
            KLLoss_positive = self.vae_sampler.kl_div(mu_positive_prior, logvar_positive_prior)
            KLLoss_negative = self.vae_sampler.kl_div(mu_negative_prior, logvar_negative_prior)
            KLLoss = KLLoss_positive + KLLoss_negative
        # Calculate KL weight
        kl_weight = self.get_kl_weight(iter)
        # Emotion processing
        if config.emo_combine == "att":
            v = self.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src)
        elif config.emo_combine == "gate":
            v = self.cdecoder(m_out, m_tilde_out)

        # Decode
        sos_token = torch.tensor([config.SOS_idx] * enc_batch.size(0), device=device).unsqueeze(1)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)

        # Прогоним BERT эмбеддинги через адаптер
        adapted_embed = self.bert_adapter(self.embedding(dec_batch_shift))

        pre_logit, attn_dist = self.decoder(adapted_embed, v, v, (mask_src, mask_trg))

        #pre_logit, attn_dist = self.decoder(self.embedding(dec_batch_shift), v, v, (mask_src, mask_trg))

        # Compute output distribution
        logit = self.generator(pre_logit, attn_dist, enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros)

        if (train and config.schedule > 10):
            if (random.uniform(0, 1) <= (0.0001 + (1 - 0.0001) * math.exp(-1. * iter / config.schedule))):
                config.oracle = True
            else:
                config.oracle = False

        if iter<6000 and train:
            total_loss = self.criterion1(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)) + kl_weight * KLLoss
        else:
            total_loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)) + kl_weight * KLLoss
        #total_loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)) + kl_weight * KLLoss

        # Loss computation
        if config.softmax:
            program_label = torch.tensor(batch['program_label'], device=device)
            L1_loss = nn.CrossEntropyLoss()(logit_prob, program_label)
            loss = total_loss + L1_loss
            loss_bce_program = L1_loss.item()
        else:
            loss = total_loss + nn.BCEWithLogitsLoss()(logit_prob, torch.tensor(batch['target_program'], dtype=torch.float, device=device))
            loss_bce_program = nn.BCEWithLogitsLoss()(logit_prob, torch.tensor(batch['target_program'], dtype=torch.float, device=device)).item()

        pred_program = np.argmax(logit_prob.detach().cpu().numpy(), axis=1)
        program_acc = accuracy_score(batch["program_label"], pred_program)

        if train:
            loss.backward()
            # Добавляем gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
            self.optimizer.step()
            
        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)).item()
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_bce_program, program_acc
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), loss_bce_program, program_acc

    def decoder_greedy(self, batch, max_dec_step=30, emotion_classifier='built_in'):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)

        # Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        # q_h = torch.mean(encoder_outputs, dim=1) if config.mean_query else encoder_outputs[:, 0]

        # # Emotion prediction
        # x = self.s_weight(q_h)
        # logit_prob = torch.matmul(x, self.emoji_embedding.weight.transpose(0, 1))
        q_h_max, _ = torch.max(encoder_outputs, dim=1)
        q_h_mean = torch.mean(encoder_outputs, dim=1)
        q_h = 0.6 * q_h_max + 0.4 * q_h_mean
        x = self.s_weight(q_h)
        logit_prob = torch.matmul(x, self.emoji_embedding.weight.transpose(0,1))
        emo_pred = torch.argmax(logit_prob, dim=-1)

        if emotion_classifier == "vader":
            context_emo = [self.positive_emotions[0] if d['compound'] > 0 else self.negative_emotions[0] for d in batch['context_emotion_scores']]
            context_emo = torch.tensor(context_emo, device=device)
            emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.vae_sampler(q_h, context_emo, self.emoji_embedding)
        elif emotion_classifier == "built_in":
            emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.vae_sampler(q_h, emo_pred, self.emoji_embedding)

        # Emotion processing
        m_out = self.emotion_input_encoder_1(emotions_mimic.unsqueeze(1), encoder_outputs, mask_src)
        m_tilde_out = self.emotion_input_encoder_2(emotions_non_mimic.unsqueeze(1), encoder_outputs, mask_src)

        if config.emo_combine == "att":
            v = self.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src)
        elif config.emo_combine == "gate":
            v = self.cdecoder(m_out, m_tilde_out)

        # Decode
        ys = torch.ones(1, 1, device=device).fill_(config.SOS_idx).long()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            adapted_embed = self.bert_adapter(self.embedding(ys))
            out, attn_dist = self.decoder(adapted_embed, v, v, (mask_src, mask_trg))
            logit = self.generator(out, attn_dist, enc_batch_extend_vocab, extra_zeros)
            _, next_word = torch.max(logit[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in next_word.view(-1)])
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.ones(1, 1, device=device).fill_(next_word).long()], dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for row in np.transpose(decoded_words):
            st = ' '.join([e for e in row if e != '<EOS>'])
            sent.append(st)
        return sent, batch['context_emotion_scores'][0]['compound'], int(emo_pred[0].item())


    def decoder_topk(self, batch, max_dec_step=30, emotion_classifier='built_in'):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)

        emotions = batch['program_label']

        context_emo = [self.positive_emotions[0] if d['compound'] > 0 else self.negative_emotions[0] for d in batch['context_emotion_scores']] 
        context_emo = torch.tensor(context_emo, device=device)

          ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)

        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        # q_h = torch.mean(encoder_outputs, dim=1) if config.mean_query else encoder_outputs[:, 0]

        # x = self.s_weight(q_h)
        # # Method 2 
        # logit_prob = torch.matmul(x, self.emoji_embedding.weight.transpose(0, 1))
        q_h_max, _ = torch.max(encoder_outputs, dim=1)
        q_h_mean = torch.mean(encoder_outputs, dim=1)
        q_h = 0.6 * q_h_max + 0.4 * q_h_mean
        x = self.s_weight(q_h)
        logit_prob = torch.matmul(x, self.emoji_embedding.weight.transpose(0,1))

        if emotion_classifier == "vader":
            context_emo = [self.positive_emotions[0] if d['compound'] > 0 else self.negative_emotions[0] for d in batch['context_emotion_scores']] 
            context_emo = torch.tensor(context_emo, device=device)
            emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.vae_sampler(q_h, context_emo, self.emoji_embedding)
        elif emotion_classifier is None:
            emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.vae_sampler(q_h, batch['program_label'], self.emoji_embedding)
        elif emotion_classifier == "built_in":
            emo_pred = torch.argmax(logit_prob, dim=-1)
            emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.vae_sampler(q_h, emo_pred, self.emoji_embedding)

        m_out = self.emotion_input_encoder_1(emotions_mimic.unsqueeze(1), encoder_outputs, mask_src)
        m_tilde_out = self.emotion_input_encoder_2(emotions_non_mimic.unsqueeze(1), encoder_outputs, mask_src)

        if config.emo_combine == "att":
            v = self.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src)
        elif config.emo_combine == "gate":
            v = self.cdecoder(m_out, m_tilde_out)
        elif config.emo_combine == 'vader':
            m_weight = context_emo_scores.unsqueeze(-1).unsqueeze(-1)
            m_tilde_weight = 1 - m_weight
            v = m_weight * m_weight + m_tilde_weight * m_tilde_out

        ys = torch.ones(1, 1, device=device).fill_(config.SOS_idx).long()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                adapted_embed = self.bert_adapter(self.embedding(ys))
                out, attn_dist = self.decoder(self.embedding_proj_in(adapted_embed),
                                              self.embedding_proj_in(encoder_outputs),
                                              (mask_src, mask_trg),
                                              attention_parameters)
            else:
                adapted_embed = self.bert_adapter(self.embedding(ys))
                out, attn_dist = self.decoder(adapted_embed, v, v, (mask_src, mask_trg))

            logit = self.generator(out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=3, top_p=0, filter_value=-float('Inf'))
            # Sample from the filtered distribution
            next_word = torch.multinomial(F.softmax(filtered_logit, dim=-1), 1).squeeze()
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in next_word.view(-1)])
            next_word = next_word.data.item()

            ys = torch.cat([ys, torch.ones(1, 1, device=device).fill_(next_word).long()], dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for row in np.transpose(decoded_words):
            st = ' '.join([e for e in row if e != '<EOS>'])
            sent.append(st)
        return sent


