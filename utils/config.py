import os
import logging
import argparse
import torch

# Определение индексов для специальных токенов
UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4
SYS_idx = 5
CLS_idx = 6

# Модели
bert_model = 'bert-base-uncased'
gpt2_model = 'gpt2'

# Определение устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Если работаем в Kaggle, использовать /kaggle/working/
base_path = "/kaggle/working" if os.path.exists("/kaggle/working") else "."

# Парсер аргументов
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="empathetic")

parser.add_argument("--emo_input", type=str, default="self_att", choices=["self_att", "cross_att"])  # cross_att; self_att
parser.add_argument("--emo_combine", type=str, default="gate", choices=["gate", "att"])  # att; gate
parser.add_argument("--decoder", type=str, default="single")  # single
parser.add_argument("--saved_model_path", type=str, default=None)  # Deprecated, use save_path instead
parser.add_argument("--vae", type=bool, default=False)  # Использовать ли VAE для случайности и добавления потерь VAE
parser.add_argument("--eq6_loss", type=bool, default=False)
parser.add_argument("--vader_loss", type=bool, default=False)  # Добавить потери VADER
parser.add_argument("--init_emo_emb", action="store_true")

parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--save_path", type=str, default=os.path.join(base_path, "save/test/"))
parser.add_argument("--save_path_dataset", type=str, default=os.path.join(base_path, "save/"))
parser.add_argument("--cuda", action="store_true", default=True)

parser.add_argument("--pointer_gen", action="store_true")
parser.add_argument("--oracle", action="store_true")
parser.add_argument("--basic_learner", action="store_true", default=True)
parser.add_argument("--project", action="store_true")
parser.add_argument("--topk", type=int, default=0)
parser.add_argument("--l1", type=float, default=0.0)
parser.add_argument("--softmax", action="store_true", default=True)
parser.add_argument("--mean_query", action="store_true")
parser.add_argument("--schedule", type=float, default=2000)

parser.add_argument("--large_decoder", action="store_true")
parser.add_argument("--multitask", action="store_true")
parser.add_argument("--is_coverage", action="store_true")
parser.add_argument("--use_oov_emb", action="store_true")
parser.add_argument("--pretrain_emb", action="store_true", default=True)
parser.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str, default="mimic")
parser.add_argument("--weight_sharing", action="store_true")
parser.add_argument("--label_smoothing", action="store_true", default=True)
parser.add_argument("--noam", action="store_true", default=False)
parser.add_argument("--universal", action="store_true")
parser.add_argument("--act", action="store_true", default=True)
parser.add_argument("--act_loss_weight", type=float, default=0.01)

parser.add_argument("--emb_file", type=str)

## Параметры трансформера
parser.add_argument("--hop", type=int, default=1)
parser.add_argument("--heads", type=int, default=2)
parser.add_argument("--depth", type=int, default=40)
parser.add_argument("--filter", type=int, default=50)
parser.add_argument("--input_dropout", type=float, default=0.1)
parser.add_argument("--layer_dropout", type=float, default=0.1)
parser.add_argument("--attention_dropout", type=float, default=0.1)
parser.add_argument("--relu_dropout", type=float, default=0.1)

def print_opts(opts):
    """Выводит значения всех аргументов командной строки."""
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

# Парсинг аргументов
args = parser.parse_args()
print_opts(args)

model = args.model
dataset = args.dataset
large_decoder = args.large_decoder
topk = args.topk
l1 = args.l1
oracle = args.oracle
basic_learner = args.basic_learner
multitask = args.multitask
softmax = args.softmax
mean_query = args.mean_query
schedule = args.schedule
input_dropout = args.input_dropout
layer_dropout = args.layer_dropout
attention_dropout = args.attention_dropout
relu_dropout = args.relu_dropout

# Гиперпараметры
hidden_dim = args.hidden_dim
emb_dim = args.emb_dim
batch_size = args.batch_size
lr = args.lr
beam_size = args.beam_size
project = args.project
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = args.max_grad_norm

# Наши аргументы
emo_input = args.emo_input
emo_combine = args.emo_combine
decoder = args.decoder
saved_model_path = args.saved_model_path
vae = args.vae
eq6_loss = args.eq6_loss
vader_loss = args.vader_loss
init_emo_emb = args.init_emo_emb

pointer_gen = args.pointer_gen
is_coverage = args.is_coverage
use_oov_emb = args.use_oov_emb
cov_loss_wt = 1.0
lr_coverage = 0.15
eps = 1e-12
epochs = 10000

emb_file = args.emb_file or f"vectors/cc.ru.{emb_dim}.vec"
pretrain_emb = args.pretrain_emb

save_path = args.save_path
save_path_dataset = args.save_path_dataset

test = args.test

### Параметры трансформера
hop = args.hop
heads = args.heads
depth = args.depth
filter = args.filter

label_smoothing = args.label_smoothing
weight_sharing = args.weight_sharing
noam = args.noam
universal = args.universal
act = args.act
act_loss_weight = args.act_loss_weight

if test:
    emo_input = 'self_att'
    emo_combine = 'gate'
    model = 'mimic'
    label_smoothing = True
    noam = args.noam
    emb_dim = 300
    emb_file = args.emb_file or f"vectors/cc.ru.{emb_dim}.vec"
    hidden_dim = 300
    hop = 1
    head = 2
    topk = 5
    pretrain_emb = False
    softmax = True
    basic_learner = True
    schedule = 10000
    saved_model_path = args.saved_model_path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
collect_stats = False

USE_CUDA = torch.cuda.is_available() and args.cuda
