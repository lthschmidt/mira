import torch
import numpy as np
from utils import config
import torch.nn.functional as F

class Beam:
    """Beam search."""

    def __init__(self, size, device=False):
        self.size = size
        self._done = False

        # The score for each translation on the beam.
        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [torch.full((size,), config.PAD_idx, dtype=torch.long, device=device)]
        self.next_ys[0][0] = config.SOS_idx

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self, word_prob):
        """Update beam status and check if finished or not."""
        num_words = word_prob.size(1)

        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]

        flat_beam_lk = beam_lk.view(-1)

        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 1st sort
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)  # 2nd sort

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # bestScoresId is flattened as a (beam x word) array,
        # so we need to calculate which word and beam each score came from
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)

        # End condition is when top-of-beam is EOS.
        if self.next_ys[-1][0].item() == config.EOS_idx:
            self._done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        """Sort the scores."""
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        """Get the score of the best in the beam."""
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        """Get the decoded sequence for the current timestep."""
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[config.SOS_idx] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        """Walk back to construct the full hypothesis."""
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x: x.item(), hyp[::-1]))


class Translator:
    """Load with trained model and handle the beam search."""

    def __init__(self, model, vocab_size):
        self.model = model
        self.vocab_size = vocab_size
        self.beam_size = config.beam_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def beam_search(self, src_seq, max_dec_step, emotion_classifier='built_in'):
        """Translation work in one batch."""

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            """Indicate the position of an instance in a tensor."""
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
            """Collect tensor parts associated to active instances."""
            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * n_bm, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(src_seq, encoder_db, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            """Collect active information."""
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)

            active_encoder_db = None

            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_encoder_db, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, enc_output, v, inst_idx_to_position_map, n_bm, enc_batch_extend_vocab, extra_zeros, mask_src, encoder_db, mask_transformer_db, DB_ext_vocab_batch):
            """Decode and update beam status, and then return active beam idx."""

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                """Prepare beam decoding sequence."""
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
                """Prepare beam decoding positions."""
                dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
                dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
                return dec_partial_pos

            def predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm, enc_batch_extend_vocab, extra_zeros, mask_src, encoder_db, mask_transformer_db, atten):
                """Predict the next word."""
                mask_trg = dec_seq.data.eq(config.PAD_idx).unsqueeze(1)
                mask_src_with_emo = torch.cat([torch.zeros((mask_src.shape[0], 1, 1), dtype=torch.bool, device=self.device), mask_src], dim=2)

                mask_src = torch.cat([mask_src[0].unsqueeze(0)] * mask_trg.size(0), 0)
                adapted_embed = self.model.bert_adapter(self.model.embedding(dec_seq))
                if config.decoder == 'mul':
                    dec_output, attn_dist = self.model.decoder(adapted_embed, enc_output, (mask_src, mask_trg), atten)
                elif config.decoder == 'single':
                    dec_output, attn_dist = self.model.decoder(adapted_embed, v, v, (mask_src, mask_trg))

                db_dist = None

                prob = self.model.generator(dec_output, attn_dist, enc_batch_extend_vocab, extra_zeros, 1, True, attn_dist_db=db_dist)
                word_prob = prob[:, -1]
                word_prob = word_prob.view(n_active_inst, n_bm, -1)
                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                """Collect active instance indices."""
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]
                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)
            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
            word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, n_active_inst, n_bm, enc_batch_extend_vocab, extra_zeros, mask_src, encoder_db, mask_transformer_db, atten=self.attention_parameters)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            """Collect hypotheses and scores."""
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        with torch.no_grad():
            # Encode
            batch = src_seq
            enc_batch, _, _, enc_batch_extend_vocab, extra_zeros, _, _ = get_input_from_batch(src_seq)
            if config.model != 'mimic':
                mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
                if config.dataset == "empathetic":
                    emb_mask = self.model.embedding(src_seq["mask_input"])
                    src_enc = self.model.encoder(self.model.embedding(enc_batch) + emb_mask, mask_src)
                else:
                    src_enc = self.model.encoder(self.model.embedding(enc_batch), mask_src)
            else:
                emotions = batch['program_label']

                context_emo = [self.model.positive_emotions[0] if d['compound'] > 0 else self.model.negative_emotions[0] for d in batch['context_emotion_scores']]
                context_emo = torch.tensor(context_emo, device=self.device)

                if config.noam:
                    self.model.optimizer.optimizer.zero_grad()
                else:
                    self.model.optimizer.zero_grad()

                mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)

                if config.dataset == "empathetic":
                    emb_mask = self.model.embedding(batch["mask_input"])
                    encoder_outputs = self.model.encoder(self.model.embedding(enc_batch) + emb_mask, mask_src)
                else:
                    encoder_outputs = self.model.encoder(self.model.embedding(enc_batch), mask_src)

                # q_h = torch.mean(encoder_outputs, dim=1) if config.mean_query else encoder_outputs[:, 0]

                # x = self.model.s_weight(q_h)
                # logit_prob = torch.softmax(torch.matmul(x, self.model.emoji_embedding.weight.transpose(0, 1)), dim=-1)

                q_h_max, _ = torch.max(encoder_outputs, dim=1)
                q_h_mean = torch.mean(encoder_outputs, dim=1)
                q_h = 0.6 * q_h_max + 0.4 * q_h_mean
                x = self.model.s_weight(q_h)
                logit_prob = torch.matmul(x, self.model.emoji_embedding.weight.transpose(0,1))
                emo_pred = torch.argmax(logit_prob, dim=-1)

                if emotion_classifier == "vader":
                    context_emo = [self.model.positive_emotions[0] if d['compound'] > 0 else self.model.negative_emotions[0] for d in batch['context_emotion_scores']]
                    context_emo = torch.tensor(context_emo, device=self.device)
                    emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.model.vae_sampler(q_h, context_emo, self.model.emoji_embedding)
                elif emotion_classifier is None:
                    emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.model.vae_sampler(q_h, batch['program_label'], self.model.emoji_embedding)
                elif emotion_classifier == "built_in":
                    emotions_mimic, emotions_non_mimic, mu_p, logvar_p, mu_n, logvar_n = self.model.vae_sampler(q_h, emo_pred, self.model.emoji_embedding)

                m_out = self.model.emotion_input_encoder_1(emotions_mimic.unsqueeze(1), encoder_outputs, mask_src)
                m_tilde_out = self.model.emotion_input_encoder_2(emotions_non_mimic.unsqueeze(1), encoder_outputs, mask_src)

                if config.emo_combine == "att":
                    v = self.model.cdecoder(encoder_outputs, m_out, m_tilde_out, mask_src)
                elif config.emo_combine == "gate":
                    v = self.model.cdecoder(m_out, m_tilde_out)
                elif config.emo_combine == 'vader':
                    m_weight = context_emo.unsqueeze(-1).unsqueeze(-1)
                    m_tilde_weight = 1 - m_weight
                    v = m_weight * m_out + m_tilde_weight * m_tilde_out
                src_enc = encoder_outputs

            encoder_db = None
            mask_transformer_db = None
            DB_ext_vocab_batch = None

            x = self.model.s_weight(q_h)
            logit_prob = torch.softmax(torch.matmul(x, self.model.emoji_embedding.weight.transpose(0, 1)), dim=-1)

            if config.topk > 0:
                k_max_value, k_max_index = torch.topk(logit_prob, config.topk)
                a = np.empty([logit_prob.shape[0], self.model.decoder_number])
                a.fill(float('-inf'))
                mask = torch.tensor(a, device=self.device)
                mask = mask.float()
                logit_prob = mask.scatter_(1, k_max_index.long(), k_max_value)

            attention_parameters = self.model.attention_activation(logit_prob)

            if config.oracle:
                attention_parameters = self.model.attention_activation(torch.tensor(src_seq['target_program'], dtype=torch.float32, device=self.device) * 1000)
            self.attention_parameters = attention_parameters.unsqueeze(-1).unsqueeze(-1)

            # Repeat data for beam search
            n_bm = self.beam_size
            n_inst, len_s, d_h = src_enc.size()
            _, self.len_program, _, _ = self.attention_parameters.size()
            src_seq = enc_batch.repeat(1, n_bm).view(n_inst * n_bm, len_s)
            src_enc = src_enc.repeat(1, n_bm, 1).view(n_inst * n_bm, len_s, d_h)

            # Prepare beams
            inst_dec_beams = [Beam(n_bm, device=self.device) for _ in range(n_inst)]

            # Bookkeeping for active or not
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            # Decode
            for len_dec_seq in range(1, max_dec_step + 1):
                active_inst_idx_list = beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, src_enc, v, inst_idx_to_position_map, n_bm, enc_batch_extend_vocab, extra_zeros, mask_src, encoder_db, mask_transformer_db, DB_ext_vocab_batch)

                if not active_inst_idx_list:
                    break  # all instances have finished their path to <EOS>

                src_seq, encoder_db, src_enc, inst_idx_to_position_map = collate_active_info(src_seq, encoder_db, src_enc, inst_idx_to_position_map, active_inst_idx_list)

        batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)

        ret_sentences = []
        for d in batch_hyp:
            ret_sentences.append(' '.join([self.model.vocab.index2word[idx] for idx in d[0]]).replace('EOS', ''))

        return ret_sentences


def sequence_mask(sequence_length, max_len=None):
    """Create a sequence mask."""
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len, dtype=torch.long, device=sequence_length.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def get_input_from_batch(batch):
    """Extract necessary inputs from the batch."""
    enc_batch = batch["input_batch"]
    enc_lens = batch["input_lengths"]
    batch_size, max_enc_len = enc_batch.size()
    assert enc_lens.size(0) == batch_size

    # Создание маски для padding
    enc_padding_mask = sequence_mask(enc_lens, max_len=max_enc_len).float()

    extra_zeros = None
    enc_batch_extend_vocab = None

    # Если используется pointer-generator, добавляем extended vocabulary
    if config.pointer_gen:
        enc_batch_extend_vocab = batch["input_ext_vocab_batch"]
        if batch["max_art_oovs"] > 0:
            extra_zeros = torch.zeros((batch_size, batch["max_art_oovs"]), device=enc_batch.device)

    # Инициализация состояния с нулями
    c_t_1 = torch.zeros((batch_size, 2 * config.hidden_dim), device=enc_batch.device)

    # Инициализация coverage (если используется)
    coverage = None
    if config.is_coverage:
        coverage = torch.zeros_like(enc_batch, device=enc_batch.device)

    return enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage
