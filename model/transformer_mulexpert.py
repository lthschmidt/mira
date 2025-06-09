import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import os
import pprint
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm, _gen_bias_mask, _gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask, get_input_from_batch, get_output_from_batch, top_k_top_p_filtering
from utils import config

pp = pprint.PrettyPrinter(indent=1)

# Определяем устройство автоматически
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
             filter_size, max_length=1000, input_dropout=0.1, layer_dropout=0.1,
             attention_dropout=0.1, relu_dropout=0.1, use_mask=False, universal=False):
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size).to(device)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size).to(device)

        params = (hidden_size,
                total_key_depth or hidden_size,
                total_value_depth or hidden_size,
                filter_size,
                num_heads,
                _gen_bias_mask(max_length) if use_mask else None,
                layer_dropout,
                attention_dropout,
                relu_dropout)

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if self.universal:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)  # ← теперь всё ок

        if config.act:
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None


    def forward(self, inputs, mask):
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            for i in range(self.num_layers):
                x = self.enc[i](x, mask)

            y = self.layer_norm(x)
        return y


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.1, layer_dropout=0.1,
                 attention_dropout=0.1, relu_dropout=0.1, universal=False):
        super(Decoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size).to(device)

        if self.universal:
            self.position_signal = _gen_timing_signal(num_layers, hidden_size).to(device)

        self.mask = _get_attn_subsequent_mask(max_length).to(device)

        params = (hidden_size,
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size,
                 num_heads,
                 _gen_bias_mask(max_length),  # mandatory
                 layer_dropout,
                 attention_dropout,
                 relu_dropout)

        if self.universal:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for _ in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, encoder_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        x = self.input_dropout(inputs)
        if not config.project:
            x = self.embedding_proj(x)

        if self.universal:
            if config.act:
                x, attn_dist, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal, self.position_signal, self.num_layers, encoder_output, decoding=True)
                y = self.layer_norm(x)
            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))
                y = self.layer_norm(x)
        else:
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))
            y = self.layer_norm(y)
        return y, attn_dist


class MulDecoder(nn.Module):
    def __init__(self, expert_num, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.1, layer_dropout=0.1,
                 attention_dropout=0.1, relu_dropout=0.1):
        super(MulDecoder, self).__init__()
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size).to(device)
        self.mask = _get_attn_subsequent_mask(max_length).to(device)

        params = (hidden_size,
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size,
                 num_heads,
                 _gen_bias_mask(max_length),  # mandatory
                 layer_dropout,
                 attention_dropout,
                 relu_dropout)

        if config.basic_learner:
            self.basic = DecoderLayer(*params)
        self.experts = nn.ModuleList([DecoderLayer(*params) for _ in range(expert_num)])
        self.dec = nn.Sequential(*[DecoderLayer(*params) for _ in range(num_layers)])

        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout=0.1)

    def forward(self, inputs, encoder_output, mask, attention_epxert):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)], 0)
        x = self.input_dropout(inputs)
        if not config.project:
            x = self.embedding_proj(x)
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
        expert_outputs = []
        if config.basic_learner:
            basic_out, _, attn_dist, _ = self.basic((x, encoder_output, [], (mask_src, dec_mask)))

        if attention_epxert.shape[0] == 1 and config.topk > 0:
            for i, expert in enumerate(self.experts):
                if attention_epxert[0, i] > 0.0001:  # Speed up inference
                    expert_out, _, attn_dist, _ = expert((x, encoder_output, [], (mask_src, dec_mask)))
                    expert_outputs.append(attention_epxert[0, i] * expert_out)
            x = torch.stack(expert_outputs, dim=1)
            x = x.sum(dim=1)
        else:
            for i, expert in enumerate(self.experts):
                expert_out, _, attn_dist, _ = expert((x, encoder_output, [], (mask_src, dec_mask)))
                expert_outputs.append(expert_out)
            x = torch.stack(expert_outputs, dim=1)  # (batch_size, expert_number, len, hidden_size)
            x = attention_epxert * x
            x = x.sum(dim=1)  # (batch_size, len, hidden_size)
        if config.basic_learner:
            x += basic_out
        y, _, attn_dist, _ = self.dec((x, encoder_output, [], (mask_src, dec_mask)))

        y = self.layer_norm(y)
        return y, attn_dist


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.p_gen_linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False, attn_dist_db=None):
        if config.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)
        logit = self.proj(x)
        if config.pointer_gen:
            vocab_dist = F.softmax(logit / temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist / temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)] * x.size(1), 1)  # Extend for all seq
            if beam_search:
                enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab_[0].unsqueeze(0)] * x.size(0), 0)  # Extend for all seq
            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_))
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class Transformer_experts(nn.Module):
    def __init__(self, vocab, decoder_number, model_file_path=None, is_eval=False, load_optim=False):
        super(Transformer_experts, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                               total_key_depth=config.depth, total_value_depth=config.depth,
                               filter_size=config.filter, universal=config.universal)
        self.decoder_number = decoder_number
        self.decoder = MulDecoder(decoder_number, config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads,
                                  total_key_depth=config.depth, total_value_depth=config.depth,
                                  filter_size=config.filter)

        self.decoder_key = nn.Linear(config.hidden_dim, decoder_number, bias=False)

        self.generator = Generator(config.hidden_dim, self.vocab_size)
        self.emoji_embedding = nn.Linear(64, config.emb_dim, bias=False)

        if config.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if config.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)

        if config.softmax:
            self.attention_activation = nn.Softmax(dim=1)
        else:
            self.attention_activation = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if config.noam:
            self.optimizer = NoamOpt(config.hidden_dim, 1, 8000,
                                    torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        if model_file_path is not None:
            print("Loading weights...")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.decoder_key.load_state_dict(state['decoder_key_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            if load_optim:
                self.optimizer.load_state_dict(state['optimizer'])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g, f1_b, ent_g, ent_b):
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'decoder_key_state_dict': self.decoder_key.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir,
                                      f'model_{iter}_{running_avg_ppl:.4f}_{f1_g:.4f}_{f1_b:.4f}_{ent_g:.4f}_{ent_b:.4f}')
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, iter, train=True):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        dec_batch, _, _, _, _ = get_output_from_batch(batch)
        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        if config.dataset == "empathetic":
            emb_mask = self.embedding(batch["mask_input"])
            encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)
        else:
            encoder_outputs = self.encoder(self.embedding(enc_batch), mask_src)
            
        q_h = torch.mean(encoder_outputs, dim=1) if config.mean_query else encoder_outputs[:, 0]
        logit_prob = self.decoder_key(q_h)  # (bsz, num_experts)
        
        if config.topk > 0:
            k_max_value, k_max_index = torch.topk(logit_prob, config.topk)
            a = np.empty([logit_prob.shape[0], self.decoder_number])
            a.fill(float('-inf'))
            mask = torch.tensor(a, device=device)
            logit_prob_ = mask.scatter_(1, k_max_index.long(), k_max_value)
            attention_parameters = self.attention_activation(logit_prob_)
        else:
            attention_parameters = self.attention_activation(logit_prob)
            
        if config.oracle:
            attention_parameters = self.attention_activation(torch.tensor(batch['target_program'], dtype=torch.float32, device=device) * 1000)
        attention_parameters = attention_parameters.unsqueeze(-1).unsqueeze(-1)  # (batch_size, expert_num, 1, 1)
        
        sos_token = torch.tensor([config.SOS_idx] * enc_batch.size(0), device=device).unsqueeze(1)
        dec_batch_shift = torch.cat((sos_token, dec_batch[:, :-1]), 1)
        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        
        pre_logit, attn_dist = self.decoder(self.embedding(dec_batch_shift), encoder_outputs, (mask_src, mask_trg), attention_parameters)
        logit = self.generator(pre_logit, attn_dist, enc_batch_extend_vocab if config.pointer_gen else None, extra_zeros, attn_dist_db=None)
        
        if train and config.schedule > 10:
            if random.uniform(0, 1) <= (0.0001 + (1 - 0.0001) * math.exp(-1. * iter / config.schedule)):
                config.oracle = True
            else:
                config.oracle = False

        if config.softmax:
            loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)) + \
                   nn.CrossEntropyLoss()(logit_prob, torch.tensor(batch['program_label'], device=device))
            loss_bce_program = nn.CrossEntropyLoss()(logit_prob, torch.tensor(batch['program_label'], device=device)).item()
        else:
            loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)) + \
                   nn.BCEWithLogitsLoss()(logit_prob, torch.tensor(batch['target_program'], dtype=torch.float32, device=device))
            loss_bce_program = nn.BCEWithLogitsLoss()(logit_prob, torch.tensor(batch['target_program'], dtype=torch.float32, device=device)).item()
        
        pred_program = np.argmax(logit_prob.detach().cpu().numpy(), axis=1)
        program_acc = accuracy_score(batch["program_label"], pred_program)

        if config.label_smoothing:
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)).item()

        if train:
            loss.backward()
            self.optimizer.step()

        if config.label_smoothing:
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_bce_program, program_acc
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), loss_bce_program, program_acc

    def compute_act_loss(self, module):
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t, dim=1) / p_t.size(1)) / p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)
        q_h = torch.mean(encoder_outputs, dim=1) if config.mean_query else encoder_outputs[:, 0]
        logit_prob = self.decoder_key(q_h)
        
        if config.topk > 0:
            k_max_value, k_max_index = torch.topk(logit_prob, config.topk)
            a = np.empty([logit_prob.shape[0], self.decoder_number])
            a.fill(float('-inf'))
            mask = torch.tensor(a, device=device)
            logit_prob = mask.scatter_(1, k_max_index.long(), k_max_value)

        attention_parameters = self.attention_activation(logit_prob)
        
        if config.oracle:
            attention_parameters = self.attention_activation(torch.tensor(batch['target_program'], dtype=torch.float32, device=device) * 1000)
        attention_parameters = attention_parameters.unsqueeze(-1).unsqueeze(-1)  # (batch_size, expert_num, 1, 1)

        ys = torch.ones(1, 1, device=device).fill_(config.SOS_idx).long()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(self.embedding_proj_in(self.embedding(ys)),
                                              self.embedding_proj_in(encoder_outputs),
                                              (mask_src, mask_trg),
                                              attention_parameters)
            else:
                out, attn_dist = self.decoder(self.embedding(ys), encoder_outputs, (mask_src, mask_trg), attention_parameters)
            
            logit = self.generator(out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            _, next_word = torch.max(logit[:, -1], dim=1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in next_word.view(-1)])
            next_word = next_word.data.item()
            ys = torch.cat([ys, torch.ones(1, 1, device=device).fill_(next_word).long()], dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for row in np.transpose(decoded_words):
            st = ' '.join([e for e in row if e != '<EOS>'])
            sent.append(st)
        return sent

    def decoder_topk(self, batch, max_dec_step=30):
        enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(batch)
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_input"])
        encoder_outputs = self.encoder(self.embedding(enc_batch) + emb_mask, mask_src)

        q_h = torch.mean(encoder_outputs, dim=1) if config.mean_query else encoder_outputs[:, 0]
        logit_prob = self.decoder_key(q_h)
        
        if config.topk > 0:
            k_max_value, k_max_index = torch.topk(logit_prob, config.topk)
            a = np.empty([logit_prob.shape[0], self.decoder_number])
            a.fill(float('-inf'))
            mask = torch.tensor(a, device=device)
            logit_prob = mask.scatter_(1, k_max_index.long(), k_max_value)

        attention_parameters = self.attention_activation(logit_prob)
        
        if config.oracle:
            attention_parameters = self.attention_activation(torch.tensor(batch['target_program'], dtype=torch.float32, device=device) * 1000)
        attention_parameters = attention_parameters.unsqueeze(-1).unsqueeze(-1)  # (batch_size, expert_num, 1, 1)

        ys = torch.ones(1, 1, device=device).fill_(config.SOS_idx).long()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step + 1):
            if config.project:
                out, attn_dist = self.decoder(self.embedding_proj_in(self.embedding(ys)),
                                              self.embedding_proj_in(encoder_outputs),
                                              (mask_src, mask_trg),
                                              attention_parameters)
            else:
                out, attn_dist = self.decoder(self.embedding(ys), encoder_outputs, (mask_src, mask_trg), attention_parameters)
            
            logit = self.generator(out, attn_dist, enc_batch_extend_vocab, extra_zeros, attn_dist_db=None)
            filtered_logit = top_k_top_p_filtering(logit[:, -1], top_k=3, top_p=0, filter_value=-float('Inf'))
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

class ACT_basic(nn.Module):
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None, decoding=False):
        # Инициализация переменных
        halting_probability = torch.zeros(inputs.shape[0], inputs.shape[1], device=device)  # [B, S]
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1], device=device)  # [B, S]
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1], device=device)  # [B, S]
        previous_state = torch.zeros_like(inputs, device=device)  # [B, S, HDD]

        step = 0
        while ((halting_probability < self.threshold) & (n_updates < max_hop)).any():
            # Добавляем временные и позиционные сигналы
            state = state + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            state = state + pos_enc[:, step, :].unsqueeze(1).repeat(1, inputs.shape[1], 1).type_as(inputs.data)

            # Вычисляем вероятность остановки
            p = self.sigma(self.p(state)).squeeze(-1)
            still_running = (halting_probability < 1.0).float()
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Обновляем вероятности и счетчики
            halting_probability = halting_probability + p * still_running
            remainders = remainders + new_halted * (1 - halting_probability)
            halting_probability = halting_probability + new_halted * remainders
            n_updates = n_updates + still_running + new_halted
            update_weights = p * still_running + new_halted * remainders

            # Применяем функцию fn (в зависимости от режима декодирования)
            if decoding:
                state, _, attention_weight = fn((state, encoder_output, []))
            else:
                state = fn(state)

            # Обновляем предыдущее состояние
            previous_state = (state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1)))

            # Обновляем предыдущие веса внимания (если декодирование)
            if decoding:
                if step == 0:
                    previous_att_weight = torch.zeros_like(attention_weight, device=device)  # [B, S, src_size]
                previous_att_weight = (attention_weight * update_weights.unsqueeze(-1)) + (previous_att_weight * (1 - update_weights.unsqueeze(-1)))

            step += 1

        # Возвращаем результат в зависимости от режима декодирования
        if decoding:
            return previous_state, previous_att_weight, (remainders, n_updates)
        else:
            return previous_state, (remainders, n_updates)

