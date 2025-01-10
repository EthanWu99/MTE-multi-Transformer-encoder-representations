import pickle
import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from scipy.stats import hmean
from prepare_dcase2024 import get_dcase2024
from audio_dataset import AudioDataset
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from transformers import HubertForSequenceClassification, WavLMForSequenceClassification, \
    UniSpeechSatForSequenceClassification, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch.optim as optim
import argparse
from beats.BEATs import BEATs, BEATsConfig
from loss import SCAdaCos
import matplotlib.pyplot as plt
from matplotlib import gridspec

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="beats", help='ast or beats')
parser.add_argument('--save_submit', type=str, default="beats_1", help='submit file name')
parser.add_argument('--train_pct', type=float, default=1, help='path to save results')
parser.add_argument('--pretrained_model_dir', type=str, default="./model", help='path to saved models')
parser.add_argument('--dataset_name', type=str, default='dcase2024')
parser.add_argument('--eval_split', type=str, default="test", help='Include test set')
parser.add_argument('--train_split', type=str, default="add", help='train, add, or train+add')
parser.add_argument('--top_k', type=int, default=1, help='Top k')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--temporal_pooling', action="store_true", default=True, help='Temporal pooling or not')
parser.add_argument('--spectral_pooling', action="store_true", default=False, help='Spectral pooling or not')
parser.add_argument('--audio_length', type=int, default=160000, help='Audio length')
parser.add_argument('--save_path', type=str, default='train_features', help='path to save results')
parser.add_argument('--n_mix_support', type=int, default=990, help='Number of support samples to mix')
parser.add_argument('--alpha', type=float, default=0.90, help='Alpha value for mixup')
parser.add_argument('--save_official', action="store_true", default=True, help='Save official submission files')

args = parser.parse_args()
model_name = args.model_name
pt_model_dir = args.pretrained_model_dir
if model_name.lower() == "beats":
    single_best_layer = 10
else:
    single_best_layer = 10

dataset_name = args.dataset_name
train_split = args.train_split
top_k = args.top_k
batch_size = args.batch_size
pooling_feature = None
temporal_pooling = args.temporal_pooling
spectral_pooling = args.spectral_pooling
if temporal_pooling:
    pooling_feature = "temporal"
elif spectral_pooling:
    pooling_feature = "spectral"
eval_split = args.eval_split
audio_length = args.audio_length
n_mix_support = args.n_mix_support
alpha = args.alpha
save_official = args.save_official

log_condition = f'MTE_model_name{model_name}_topk{top_k}_trainsplit{train_split}_evalsplit{eval_split}_pooling{pooling_feature}_n_mix_support{n_mix_support}_alpha{alpha}'
save_dir = f"out/{dataset_name}_test/{model_name}/{log_condition}/log_{time.strftime('%Y%m%d-%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

datasets = get_dcase2023(train_split, dataset_name)
train_data = datasets['train']
test_data = datasets[eval_split]

train_file_attrs = np.array(train_data['file_attrs'])
train_machine_names = np.array(train_data['machine_names'])


class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentiveStatisticsPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (batch_size, time_steps, input_dim)
        attention_weights = self.attention(x)  # (batch_size, time_steps, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch_size, time_steps, 1)

        # 加权均值
        mean = torch.sum(attention_weights * x, dim=1)  # (batch_size, input_dim)

        # 加权标准差
        variance = torch.sum(attention_weights * (x - mean.unsqueeze(1)) ** 2, dim=1)  # (batch_size, input_dim)
        std_dev = torch.sqrt(variance + 1e-12)  # 避免开方时的数值不稳定性

        # 连接均值和标准差
        utterance_embedding = torch.cat([mean, std_dev], dim=1)  # (batch_size, 2 * input_dim)

        return utterance_embedding


class BEATsWithNewLayer(BEATs):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.asp_layer = ECAPA_TDNN.AttentiveStatisticsPooling(768,global_context=False)
        self.asp_layer = AttentiveStatisticsPooling(768)
        self.layer1 = nn.Linear(768 * 2, 1024, bias=False)  # 添加新的全连接层，假设你需要从527映射到202
        self.layer2 = nn.Linear(1024, 154, bias=False)  # 添加新的全连接层，假设你需要从527映射到202
        self.scadacos = SCAdaCos(n_classes=154, n_subclusters=16, trainable=True)

    def forward(self, source, label, padding_mask=None):
        x, _, lprobs = self.extract_features(source, padding_mask, need_weights=True, layer=11)
        # utterance_embedding = self.asp_layer(x.transpose(1, 2)).transpose(1, 2).squeeze(1)
        utterance_embedding = self.asp_layer(x)
        logit = self.layer1(utterance_embedding)
        logit = self.layer2(logit)
        # logit = self.scadacos((lprobs, label, label))
        return logit, lprobs


class CustomHubertForSequenceClassification(HubertForSequenceClassification):
    def __init__(self, config, num_labels=154):
        super().__init__(config)
        self.num_labels = num_labels

        self.classifier = nn.Linear(256, self.num_labels)

        self.init_weights()

    def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None,
                return_dict=None):

        outputs = self.hubert(input_values, attention_mask=attention_mask, output_attentions=output_attentions,
                              output_hidden_states=output_hidden_states, return_dict=True)
        hidden_states = outputs.last_hidden_state


        penultimate_layer_output = self.projector(hidden_states[:, 0, :])


        logits = self.classifier(penultimate_layer_output)

        return logits, hidden_states


class CustomUniSpeechSatForSequenceClassification(UniSpeechSatForSequenceClassification):
    def __init__(self, config, num_labels=154):
        super().__init__(config)
        self.num_labels = num_labels


        self.classifier = nn.Linear(256, self.num_labels)


        self.init_weights()

    def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None,
                return_dict=None):

        outputs = self.unispeech_sat(input_values, attention_mask=attention_mask, output_attentions=output_attentions,
                                     output_hidden_states=output_hidden_states, return_dict=True)
        hidden_states = outputs.last_hidden_state

        penultimate_layer_output = self.projector(hidden_states[:, 0, :])

        logits = self.classifier(penultimate_layer_output)

        return logits, penultimate_layer_output


class CustomWav2VecForSequenceClassification(Wav2Vec2ForSequenceClassification):
    def __init__(self, config, num_labels=154):
        super().__init__(config)
        self.num_labels = num_labels

        self.classifier = nn.Linear(256, self.num_labels)

        self.init_weights()

    def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, labels=None):

        outputs = self.wav2vec2(input_values, attention_mask=attention_mask, output_attentions=output_attentions,
                                output_hidden_states=output_hidden_states, return_dict=True)
        hidden_states = outputs.last_hidden_state  # 获取最后一层的隐藏状态


        penultimate_layer_output = self.projector(hidden_states[:, 0, :])  # 仅使用 [CLS] token 的输出


        logits = self.classifier(penultimate_layer_output)

        return logits, penultimate_layer_output


class CustomWavLMForSequenceClassification(WavLMForSequenceClassification):
    def __init__(self, config, num_labels=154):
        super().__init__(config)
        self.num_labels = num_labels

        self.classifier = nn.Linear(256, self.num_labels)


        self.init_weights()

    def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None,
                return_dict=None):

        outputs = self.wavlm(input_values, attention_mask=attention_mask, output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, return_dict=True)
        hidden_states = outputs.last_hidden_state  # 获取最后一层的隐藏状态

        penultimate_layer_output = self.projector(hidden_states[:, 0, :])  # 仅使用 [CLS] token 的输出

        logits = self.classifier(penultimate_layer_output)

        return logits, outputs

model = torch.load('./model/BEATs_MTE_Warmup_Mixup0.9_1.h5')


# ======== distance matrix ========
# original
# def calc_dist_matrix(x, y):
#     """Calculate Euclidean distance matrix with torch.tensor"""
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     x = x.unsqueeze(1).expand(n, m, d)
#     y = y.unsqueeze(0).expand(n, m, d)
#     dist_matrix = torch.sqrt(torch.pow(x - y, 2).sum(2))
#     return dist_matrix

def calc_dist_matrix(x, y, bs=16):
    """Calculate Euclidean distance matrix with torch.tensor"""
    if bs > x.shape[0]:
        bs = x.shape[0]
    dist_matrices = torch.zeros(x.size(0), y.size(0))
    d = x.size(1)
    for i in range(y.shape[0]):
        m = 1
        dist_x_list = torch.zeros(x.size(0))
        x_batch = x.size(0) // bs + 1 if x.size(0) % bs != 0 else x.size(0) // bs
        for j in range(x_batch):
            x_i = x[j * bs:j * bs + bs]
            n = x_i.size(0)
            x_i = x_i.unsqueeze(1).expand(n, m, d)  # 16, 1, 768
            y_i = y[i].unsqueeze(0).expand(n, m, d)
            dist_matrix = torch.sqrt(torch.pow(x_i - y_i, 2).sum(2))
            dist_x_list[j * bs:j * bs + bs] = dist_matrix.reshape(-1)
        dist_matrices[:, i] = dist_x_list
    return dist_matrices


# ======== evaluation function ========
def eval_score(gt_list, scores):
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, scores)
    img_roc_auc = roc_auc_score(gt_list, scores)
    pauc = roc_auc_score(gt_list, scores, max_fpr=0.1)
    precision, recall, thresholds = precision_recall_curve(gt_list, scores)
    f1_scores = (2 * precision * recall) / (precision + recall + np.finfo(float).eps)
    idx = np.argmax(f1_scores)
    return img_roc_auc, pauc, f1_scores[idx]


device = "cuda"
model.to(device)
model.eval()

df_log = pd.DataFrame()
df_log_wo_norm = pd.DataFrame()

from sklearn.preprocessing import LabelEncoder

# ======= Evaluation =======
machine_names = np.unique(train_data["machine_names"])
print(f'machine names: {machine_names}')

for class_name in machine_names:

    train_dataset = AudioDataset(
        file_list=train_data["file_list"],
        label_list=train_data["label_list"],
        machine_list=train_data["machine_names"],
        source_list=train_data["source_list"],
        machine_name=class_name,
        audio_length=audio_length,
    )
    test_dataset = AudioDataset(
        file_list=test_data["file_list"],
        label_list=test_data["label_list"],
        machine_list=test_data["machine_names"],
        source_list=test_data["source_list"],
        machine_name=class_name,
        audio_length=audio_length,
    )
    train_class_ids_machine = train_file_attrs[train_machine_names == class_name]
    all_attrs = np.concatenate([train_class_ids_machine])
    label_encoder = LabelEncoder()
    label_encoder.fit(all_attrs)
    train_class_ids = label_encoder.transform(train_class_ids_machine)

    class_tensor = torch.from_numpy(train_class_ids)
    unique_class_tensor = torch.unique(class_tensor)
    # print(f'unique class tensor: {unique_class_tensor}')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=False)
    print(f'class name: {class_name}')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=False)

    # extract train set features
    print(f'len train dataset: {len(train_dataset)}')
    train_feature_layers = []
    train_all_lasts = []

    # save the embeddings if None
    save_path = args.save_path + f'_{model_name}'
    save_dir_path = f"temp_{dataset_name}_{train_split}"
    os.makedirs(os.path.join(save_path, save_dir_path), exist_ok=True)
    train_feature_filepath = os.path.join(save_path, save_dir_path, f'train_{pooling_feature}_%s.pkl' % class_name)
    label = np.zeros((batch_size, 154))
    label = torch.tensor(label)
    label = label.to(device)
    if not os.path.exists(train_feature_filepath):
        for batch in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            x = batch['input']
            x = x.to(device)
            bs = x.size(0)
            padding_mask = torch.zeros(x.shape).bool()
            padding_mask = padding_mask.to(device)

            # forward
            with torch.no_grad():
                _, attns = model(source=x, label=label)

            # pick pooling technique
            if temporal_pooling:
                train_out_layers = [f_layer[0].transpose(0, 1).reshape(bs, 62, 8, -1).mean(1).cpu().unsqueeze(0) for
                                    f_layer in attns][1:]  # skip the first pass: 12, bs, 496, 768 -> 12, bs, 768
            elif spectral_pooling:
                train_out_layers = [f_layer[0].transpose(0, 1).reshape(bs, 62, 8, -1).mean(2).cpu().unsqueeze(0) for
                                    f_layer in attns][1:]  # skip the first pass: 12, bs, 496, 768 -> 12, bs, 768
            else:
                train_out_layers = [f_layer[0].transpose(0, 1).mean(1).cpu().unsqueeze(0) for f_layer in attns][
                                   1:]  # skip the first pass: 12, bs, 496, 768 -> 12, bs, 768
            train_feature_layers.append(torch.cat(train_out_layers, 0))

        torch.cuda.empty_cache()
        train_feature_layers = torch.cat(train_feature_layers, 1).flatten(2)

    gt_list = []
    test_outputs = []
    test_feature_layers = []
    test_all_lasts = []

    save_dir_path = f"temp_{dataset_name}_{eval_split}"
    os.makedirs(os.path.join(save_path, save_dir_path), exist_ok=True)
    test_feature_filepath = os.path.join(save_path, save_dir_path,
                                         f'{eval_split}_{pooling_feature}_%s.pkl' % class_name)

    if not os.path.exists(test_feature_filepath):
        # extract test set features
        for batch in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            x = batch['input']
            x = x.to(device)
            bs = x.size(0)
            y = batch['label']
            gt_list.extend(y.cpu().detach().numpy())
            padding_mask = torch.zeros(x.shape).bool()
            padding_mask = padding_mask.to(device)

            # forward
            with torch.no_grad():
                _, attns = model(source=x, label=label)
            # pick pooling technique
            if temporal_pooling:
                test_out_layers = [f_layer[0].transpose(0, 1).reshape(bs, 62, 8, -1).mean(1).cpu().unsqueeze(0) for
                                   f_layer in attns][1:]  # skip the first pass: 12, bs, 496, 768 -> 12, bs, 768
            elif spectral_pooling:
                test_out_layers = [f_layer[0].transpose(0, 1).reshape(bs, 62, 8, -1).mean(2).cpu().unsqueeze(0) for
                                   f_layer in attns][1:]  # skip the first pass: 12, bs, 496, 768 -> 12, bs, 768
            else:
                test_out_layers = [f_layer[0].transpose(0, 1).mean(1).cpu().unsqueeze(0) for f_layer in attns][
                                  1:]  # skip the first pass: 12, bs, 496, 768 -> 12, bs, 768
            test_feature_layers.append(torch.cat(test_out_layers, 0))

        torch.cuda.empty_cache()

        test_feature_layers = torch.cat(test_feature_layers, 1).flatten(2)

    root_path = "./"
    if eval_split == 'test':
        gt_list = np.array(pd.read_csv(
            root_path + dataset_name + '_task2_evaluator/ground_truth_data/ground_truth_' + class_name + '_section_00_test.csv',
            header=None).iloc[:, 1] == 1)
        source_list_machine = np.array(pd.read_csv(
            root_path + dataset_name + '_task2_evaluator/ground_truth_domain/ground_truth_' + class_name + '_section_00_test.csv',
            header=None).iloc[:, 1] == 0)
    elif eval_split == 'valid':
        machine_list = np.array(test_data['machine_names'])
        source_list = np.array(test_data['source_list'])
        source_list_machine = source_list[machine_list == class_name].reshape(-1)
        source_list_machine = np.array(source_list_machine == 'source')
        gt_list = np.asarray(gt_list).reshape(-1)

    # gt_source = gt_list[(source_list_machine == True) | (gt_list != 0)]
    # gt_target = gt_list[(source_list_machine == False) | (gt_list != 0)]

    gt_source = gt_list[(source_list_machine == True)]
    gt_target = gt_list[(source_list_machine == False)]

    results_an = pd.DataFrame()
    results_an['output1'] = [f.split('/')[-1] for f in test_dataset.file_list]
    results_dec = pd.DataFrame()
    results_dec['output1'] = [f.split('/')[-1] for f in test_dataset.file_list]
    best_score = 0
    best_layer = None
    for num_layer, (train_all_lasts, test_all_lasts) in enumerate(zip(train_feature_layers, test_feature_layers)):
        print(f' layer {num_layer + 1}')

        augmented_target_samples = []

        if n_mix_support is not None:
            source_train_features = train_all_lasts[:990]
            target_train_features = train_all_lasts[990:]

            source_classes = np.unique(train_class_ids[:990])
            target_classes = np.unique(train_class_ids[990:])

            target_sep_feats = []
            target_sep_feats.append(target_train_features)

            ST_dist_matrices = []
            topk_values = []
            topk_indexes = []
            all_feats = []
            for target_feat in target_sep_feats:
                all_feat = source_train_features
                ST_dist_matrix = calc_dist_matrix(torch.flatten(target_feat, 1),
                                                  torch.flatten(all_feat, 1),
                                                  bs=32)
                topk_value, topk_index = torch.topk(ST_dist_matrix, k=n_mix_support, dim=1, largest=False)
                topk_values.append(topk_value)
                topk_indexes.append(topk_index)
                ST_dist_matrices.append(ST_dist_matrix)
                all_feats.append(all_feat)

            for ix, topk_index in enumerate(topk_indexes):
                target_feat = target_sep_feats[ix]
                for i in range(len(topk_index)):
                    for a_support_set in all_feats[ix][topk_index[i]]:
                        mixup_sample = alpha * target_feat[i] + (1 - alpha) * a_support_set
                        augmented_target_samples.append(mixup_sample)

            augmented_target_samples = torch.stack(augmented_target_samples)
            print(f"augmented_target_samples: {augmented_target_samples.size()}")

            train_all_lasts = torch.cat([train_all_lasts, augmented_target_samples], 0)

        dist_matrix = calc_dist_matrix(torch.flatten(test_all_lasts, 1),
                                       torch.flatten(train_all_lasts, 1),
                                       bs=32)

        # create source and target memory banks
        source_dist = dist_matrix[:, :990]
        target_dist = dist_matrix[:, 990:]
        print(f"source_dist.size(): {source_dist.size()}")
        print(f"target_dist.size(): {target_dist.size()}")

        # implement soft scoring
        topk_value_source, topk_index_source = torch.topk(source_dist, k=top_k, dim=1, largest=False)
        topk_value_target, topk_index_target = torch.topk(target_dist, k=top_k, dim=1, largest=False)
        topk_index_target = topk_index_target + 990

        topk_indexes = torch.cat([topk_index_source, topk_index_target], 1)
        value_source = torch.mean(topk_value_source, 1).cpu().detach().numpy()
        value_target = torch.mean(topk_value_target, 1).cpu().detach().numpy()

        # Standardize source and target scores
        source_mean = np.mean(value_source)
        source_std = np.std(value_source)
        standardized_source_scores = (value_source - source_mean) / source_std
        target_mean = np.mean(value_target)
        target_std = np.std(value_target)
        standardized_target_scores = (value_target - target_mean) / target_std

        # with norm
        # pick wheter the score from source or target
        scores = np.minimum(standardized_source_scores, standardized_target_scores)
        min_indices = np.argmin(np.stack((standardized_source_scores, standardized_target_scores), axis=-1), axis=-1)
        topk_indexes_w_norm = topk_indexes[np.arange(topk_indexes.shape[
                                                         0]), min_indices]  # since now topk_indexes are concatenated, we need to find the correct index

        # without norm
        scores_wo_norm = np.minimum(value_source, value_target)
        min_indices_wo_norm = np.argmin(np.stack((value_source, value_target), axis=-1), axis=-1)
        topk_indexes_wo_norm = topk_indexes[np.arange(topk_indexes.shape[0]), min_indices_wo_norm]

        score_source = scores[(source_list_machine == True)]
        score_target = scores[(source_list_machine == False)]

        auc_all, pauc_all, f1_all = eval_score(gt_list, scores)
        auc_source, pauc_source, f1_source = eval_score(gt_source, score_source)
        auc_target, pauc_target, f1_target = eval_score(gt_target, score_target)

        official_score = hmean([auc_source, auc_target, pauc_all])

        score_source_wo_norm = scores_wo_norm[(source_list_machine == True)]
        score_target_wo_norm = scores_wo_norm[(source_list_machine == False)]

        auc_all_wo_norm, pauc_all_wo_norm, f1_all_wo_norm = eval_score(gt_list, scores_wo_norm)
        auc_source_wo_norm, pauc_source_wo_norm, f1_source_wo_norm = eval_score(gt_source, score_source_wo_norm)
        auc_target_wo_norm, pauc_target_wo_norm, f1_target_wo_norm = eval_score(gt_target, score_target_wo_norm)

        official_score_wo_norm = hmean([auc_source_wo_norm, auc_target_wo_norm, pauc_all_wo_norm])

        print('AUC: %.4f, PAUC: %.4f, F1: %.4f, OS: %.4f, OS_noNorm: %.4f' % (
            auc_all, pauc_all, f1_all, official_score, official_score_wo_norm))
        print('AUC source: %.4f, PAUC source: %.4f, F1 source: %.4f' % (auc_source, pauc_source, f1_source))
        print('AUC target: %.4f, PAUC target: %.4f, F1 target: %.4f' % (auc_target, pauc_target, f1_target))

        # ===== find the retrieval accuracy
        # with norm
        topk_indexes_source = topk_indexes_w_norm[source_list_machine]
        topk_indexes_target = topk_indexes_w_norm[~source_list_machine]
        source = 0

        if n_mix_support is not None:
            acc_source = 0
            acc_target = 0
            acc_source_wo_norm = 0
            acc_target_wo_norm = 0
        else:
            print(f'len topk indexes source: {len(topk_indexes_source)}')
            for id, ix in enumerate(topk_indexes_source):
                ret_name = train_dataset.file_list[ix].split('/')[-1]
                # print(ix, ret_name)
                if 'source' in ret_name:
                    source += 1
            acc_source = source / len(topk_indexes_source)

            target = 0
            print(f'len topk indexes target: {len(topk_indexes_target)}')
            # print(topk_indexes_target)
            for id, ix in enumerate(topk_indexes_target):
                # print(f'ix: {ix}')
                ret_name = train_dataset.file_list[ix].split('/')[-1]
                # print(ix, ret_name)
                if 'target' in ret_name:
                    target += 1
            acc_target = target / len(topk_indexes_target)
            print(f'retrieval acc source: {acc_source}, target: {acc_target}')

            # without norm
            topk_indexes_source_wo_norm = topk_indexes_wo_norm[source_list_machine]
            topk_indexes_target_wo_norm = topk_indexes_wo_norm[~source_list_machine]
            source_wo_norm = 0
            print(f'len topk indexes source: {len(topk_indexes_source_wo_norm)}')
            for id, ix in enumerate(topk_indexes_source_wo_norm):
                ret_name = train_dataset.file_list[ix].split('/')[-1]
                # print(ix, ret_name)
                if 'source' in ret_name:
                    source_wo_norm += 1
            acc_source_wo_norm = source_wo_norm / len(topk_indexes_source_wo_norm)

            target_wo_norm = 0
            print(f'len topk indexes target: {len(topk_indexes_target_wo_norm)}')
            for id, ix in enumerate(topk_indexes_target_wo_norm):
                ret_name = train_dataset.file_list[ix].split('/')[-1]
                # print(ix, ret_name)
                if 'target' in ret_name:
                    target_wo_norm += 1
            acc_target_wo_norm = target_wo_norm / len(topk_indexes_target_wo_norm)
            print(f'retrieval acc source: {acc_source_wo_norm}, target: {acc_target_wo_norm}')

        df_log = pd.concat([df_log, pd.DataFrame({
            'layer': [num_layer + 1],
            'machine': [class_name],
            'auc_source': [auc_source],
            'auc_target': [auc_target],
            'pauc': [pauc_all],
            'official_score': [official_score],
            'auc': [auc_all],
            'pauc_source': [pauc_source],
            'pauc_target': [pauc_target],
            'acc_source': [acc_source],
            'acc_target': [acc_target],
            'f1': [f1_all],
            'f1_source': [f1_source],
            'f1_target': [f1_target],
        })]
                           )
        df_log.to_csv(f'{save_dir}/result.csv', index=False)

        df_log_wo_norm = pd.concat([df_log_wo_norm, pd.DataFrame({
            'layer': [num_layer + 1],
            'machine': [class_name],
            'auc_source': [auc_source_wo_norm],
            'auc_target': [auc_target_wo_norm],
            'pauc': [pauc_all_wo_norm],
            'official_score': [official_score_wo_norm],
            'auc': [auc_all_wo_norm],
            'pauc_source': [pauc_source_wo_norm],
            'pauc_target': [pauc_target_wo_norm],
            'acc_source': [acc_source_wo_norm],
            'acc_target': [acc_target_wo_norm],
            'f1': [f1_all_wo_norm],
            'f1_source': [f1_source_wo_norm],
            'f1_target': [f1_target_wo_norm],
        })]
                                   )
        df_log_wo_norm.to_csv(f'{save_dir}/result_wo_norm.csv', index=False)

        # ======== create challenge submission files ========
        # use layer 5 for anomaly detection
        if save_official:
            sub_path_sb = './' + dataset_name + f'_task2_evaluator/teams/submission/team_{args.save_submit}_pooling{pooling_feature}_n_mix_support{n_mix_support}_alpha{alpha}_single_best'
            sub_path_mb = './' + dataset_name + f'_task2_evaluator/teams/submission/team_{args.save_submit}_pooling{pooling_feature}_n_mix_support{n_mix_support}_alpha{alpha}_machine_best'
            if not os.path.exists(sub_path_sb):
                os.makedirs(sub_path_sb)
            if not os.path.exists(sub_path_mb):
                os.makedirs(sub_path_mb)

            results_an[num_layer + 1] = [str(s) for s in scores]
            # decision results
            precision, recall, thresholds = precision_recall_curve(gt_list, scores)
            f1_scores = (2 * precision * recall) / (precision + recall + np.finfo(float).eps)
            idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[idx]
            # threshold = np.percentile(train_scores, q=90)
            decisions = scores > optimal_threshold
            results_dec[num_layer + 1] = [str(int(s)) for s in decisions]

            if num_layer + 1 == single_best_layer:
                results_an_single = pd.DataFrame()
                results_an_single['output1'] = results_an['output1']
                results_an_single[num_layer + 1] = results_an[num_layer + 1]
                results_dec_single = pd.DataFrame()
                results_dec_single['output1'] = results_dec['output1']
                results_dec_single[num_layer + 1] = results_dec[num_layer + 1]
                results_an_single.to_csv(
                    sub_path_sb + '/anomaly_score_' + args.save_submit + class_name + '_section_00' + '_test.csv',
                    encoding='utf-8', index=False, header=False)
                results_dec_single.to_csv(
                    sub_path_sb + '/decision_result_' + args.save_submit + class_name + '_section_00' + '_test.csv',
                    encoding='utf-8', index=False, header=False)

            if official_score > best_score:
                best_score = official_score
                best_layer = num_layer + 1

    if save_official:
        # save file
        print(f'best layer: {best_layer}')
        print(f'best score: {best_score}')
        results_an_best = pd.DataFrame()
        results_an_best['output1'] = results_an['output1']
        results_an_best[best_layer] = results_an[best_layer]
        results_dec_best = pd.DataFrame()
        results_dec_best['output1'] = results_dec['output1']
        results_dec_best[best_layer] = results_dec[best_layer]
        results_an_best.to_csv(
            sub_path_mb + '/anomaly_score_' + args.save_submit + class_name + '_section_00' + '_test.csv',
            encoding='utf-8', index=False, header=False)
        results_dec_best.to_csv(
            sub_path_mb + '/decision_result_' + args.save_submit + class_name + '_section_00' + '_test.csv',
            encoding='utf-8', index=False, header=False)
        # save to DCASE evaluator path
        if not os.path.exists(sub_path_mb):
            os.makedirs(sub_path_mb)
        results_an_best.to_csv(
            sub_path_mb + '/anomaly_score_' + args.save_submit + class_name + '_section_00' + '_test.csv',
            encoding='utf-8', index=False, header=False)
        results_dec_best.to_csv(
            sub_path_mb + '/decision_result_' + args.save_submit + class_name + '_section_00' + '_test.csv',
            encoding='utf-8', index=False, header=False)

# ======== Summary with norm ========
df_test = df_log.reset_index(drop=True)
columns = df_test.columns
df_avg = df_test.groupby('layer')[columns[2:]].agg(hmean)
df_avg['oc'] = df_avg[['auc_source', 'auc_target', 'pauc']].agg(hmean, axis=1)  # just in case
print(df_avg)

print(f'BEST LAYER WISE SCORE')
best_layer_wise = df_avg['oc'].argmax() + 1  # index + 1 to get the layer
machine_layer_wise = df_test[df_test['layer'] == best_layer_wise][
    ['machine', 'auc', 'pauc', 'auc_source', 'auc_target', 'official_score']]
print(machine_layer_wise)

print(f' All layer wise score:')
final_best_layer_wise = pd.DataFrame(df_avg.iloc[df_avg['official_score'].argmax()]).T
w_office_score = final_best_layer_wise
print(final_best_layer_wise)

print(f'BEST each machine')
best_each_machine = df_test.loc[df_test.groupby('machine')['official_score'].idxmax()]
print(best_each_machine)

print(f'FINAL BEST SCORE')
final_best_score = pd.DataFrame(best_each_machine[['auc_source', 'auc_target', 'pauc', 'official_score']].agg(hmean)).T
print(final_best_score)

df_log.to_csv(f'{save_dir}/result.csv', index=False)
df_avg.to_csv(f'{save_dir}/df_avg.csv', index=False)
machine_layer_wise.to_csv(f'{save_dir}/machine_layer_wise.csv', index=False)
final_best_layer_wise.to_csv(f'{save_dir}/final_best_layer_wise.csv', index=False)
best_each_machine.to_csv(f'{save_dir}/best_each_machine.csv', index=False)
final_best_score.to_csv(f'{save_dir}/final_best_score.csv', index=False)

# ======== Summary without norm ========
df_test = df_log_wo_norm.reset_index(drop=True)
columns = df_test.columns
df_avg = df_test.groupby('layer')[columns[2:]].agg(hmean)
df_avg['oc'] = df_avg[['auc_source', 'auc_target', 'pauc']].agg(hmean, axis=1)  # just in case
print(df_avg)

print(f'BEST LAYER WISE SCORE')
best_layer_wise = df_avg['oc'].argmax() + 1  # index + 1 to get the layer
machine_layer_wise = df_test[df_test['layer'] == best_layer_wise][
    ['machine', 'auc', 'pauc', 'auc_source', 'auc_target', 'official_score']]
print(machine_layer_wise)

print(f' All layer wise score:')
final_best_layer_wise = pd.DataFrame(df_avg.iloc[df_avg['official_score'].argmax()]).T
wo_office_score = final_best_layer_wise
print(final_best_layer_wise)

print(f'BEST each machine')
best_each_machine = df_test.loc[df_test.groupby('machine')['official_score'].idxmax()]
print(best_each_machine)

print(f'FINAL BEST SCORE')
final_best_score = pd.DataFrame(best_each_machine[['auc_source', 'auc_target', 'pauc', 'official_score']].agg(hmean)).T
print(final_best_score)

df_log_wo_norm.to_csv(f'{save_dir}/result_wo_norm.csv', index=False)

print(f"Save log to {save_dir}")
df_avg.to_csv(f'{save_dir}/df_avg_wo_norm.csv', index=False)
machine_layer_wise.to_csv(f'{save_dir}/machine_layer_wise_wo_norm.csv', index=False)
final_best_layer_wise.to_csv(f'{save_dir}/final_best_layer_wise_wo_norm.csv', index=False)
best_each_machine.to_csv(f'{save_dir}/best_each_machine_wo_norm.csv', index=False)
final_best_score.to_csv(f'{save_dir}/final_best_score_wo_norm.csv', index=False)

print(w_office_score)
print(wo_office_score)
print(f"Done")
