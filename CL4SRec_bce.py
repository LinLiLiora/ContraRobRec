import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
import logging
import time as Time
from utility import pad_history, calculate_hit, extract_axis_1
from collections import Counter
from model.SASRec import *
import random
from math import floor

logging.getLogger().setLevel(logging.INFO)

'''
    1. 数据增强方法
    2. bce、bpr、mse 
    3. 优化代码
    4. 数据集
    5. 写论文 CL4SRec + DROS
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")

    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='rr',
                        help='yc, ks, rr')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='num_filters')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--model_name', type=str, default='SASRec_bce',
                        help='model name.')
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-6,
                        help='l2 loss reg coef.')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='dro alpha.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='for robust radius')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    parser.add_argument('--descri', type=str, default='',
                        help='description of the work.')
    parser.add_argument('--cl_rate', type=float, default=0.1,
                        help='cl_rate')
    parser.add_argument('--aug_type', type=int, default=0,
                        help='aug_type [0,1,2]')
    parser.add_argument('--aug_rate', type=float, default=0.2,
                        help='aug_rate')
    return parser.parse_args()


def evaluate(model, test_data, device):
    eval_sessions=pd.read_pickle(os.path.join(data_directory, test_data))
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated=0
    total_clicks=1.0
    total_purchase = 0.0
    total_reward = [0, 0, 0, 0]
    hit_clicks=[0,0,0,0]
    ndcg_clicks=[0,0,0,0]
    hit_purchase=[0,0,0,0]
    ndcg_purchase=[0,0,0,0]
    while evaluated<len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        for i in range(batch):
            id=eval_ids[evaluated]
            group=groups.get_group(id)
            history=[]
            for index, row in group.iterrows():
                state=list(history)
                state = [int(i) for i in state]
                len_states.append(seq_size if len(state)>=seq_size else 1 if len(state)==0 else len(state))
                state=pad_history(state,seq_size,item_num)
                states.append(state)
                action=row['item_id']
                try:
                    is_buy=row['t_read']
                except:
                    is_buy=row['time']
                reward = 1 if is_buy >0 else 0
                if is_buy>0:
                    total_purchase+=1.0
                else:
                    total_clicks+=1.0
                actions.append(action)
                rewards.append(reward)
                history.append(row['item_id'])
            evaluated+=1
            if evaluated >= len(eval_ids):
                break

        states = np.array(states)
        states = torch.LongTensor(states)
        states = states.to(device)

        prediction = model.forward_eval(states, np.array(len_states))
        prediction = prediction.cpu()
        prediction = prediction.detach().numpy()
        sorted_list=np.argsort(prediction)
        calculate_hit(sorted_list,topk,actions,rewards,reward_click,total_reward,hit_clicks,ndcg_clicks,hit_purchase,ndcg_purchase)
    print('#############################################################')
    hr_list = []
    ndcg_list = []
    print('hr@{}\tndcg@{}\thr@{}\tndcg@{}\thr@{}\tndcg@{}'.format(topk[0], topk[0], topk[1], topk[1], topk[2], topk[2]))
    for i in range(len(topk)):
        hr_purchase=hit_purchase[i]/total_purchase
        ng_purchase=ndcg_purchase[i]/total_purchase
        hr_list.append(hr_purchase)
        ndcg_list.append(ng_purchase[0,0])
        if i == 1:
            hr_20 = hr_purchase

    print('{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    print('{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}&{:.4f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))
    print('#############################################################')

    return hr_20


def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 @ view2.T) / temperature
    score = torch.diag(F.log_softmax(pos_score, dim=1))

    return -score.mean()


def calcu_propensity_score(buffer):
    items = list(buffer['next'])
    freq = Counter(items)
    for i in range(item_num):
        if i not in freq.keys():
            freq[i] = 0
    pop = [freq[i] for i in range(item_num)]
    pop = np.array(pop)
    ps = pop + 1
    ps = ps / np.sum(ps)
    ps = np.power(ps, 0.05)

    return ps


class SequenceAugmentor(object):
    def __init__(self):
        pass

    @staticmethod
    def item_crop(seq, seq_len, crop_ratio):
        augmented_seq = np.zeros_like(seq)
        augmented_pos = np.zeros_like(seq)
        aug_len = []
        for i, s in enumerate(seq):
            start = random.sample(range(seq_len[i]-floor(seq_len[i]*crop_ratio)),1)[0]
            crop_len = floor(seq_len[i]*crop_ratio)+1
            augmented_seq[i,:crop_len] =seq[i,start:start+crop_len]
            augmented_pos [i,:crop_len] = range(1,crop_len+1)
            aug_len.append(crop_len)
        return augmented_seq, augmented_pos, aug_len

    @staticmethod
    def item_reorder(seq, seq_len, reorder_ratio):
        augmented_seq = seq.copy()
        for i, s in enumerate(seq):
            start = random.sample(range(seq_len[i]-floor(seq_len[i]*reorder_ratio)),1)[0]
            np.random.shuffle(augmented_seq[i,start:floor(seq_len[i]*reorder_ratio)+start+1])
        return augmented_seq

    @staticmethod
    def item_mask(seq, seq_len, mask_ratio, mask_idx):
        augmented_seq = seq.copy()
        for i, s in enumerate(seq):
            to_be_masked = random.sample(range(seq_len[i]), floor(seq_len[i]*mask_ratio))
            augmented_seq[i, to_be_masked] = mask_idx
        return augmented_seq


if __name__ == '__main__':

    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    data_directory = './data/' + args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items
    reward_click = args.r_click
    reward_buy = args.r_buy
    topk=[10, 20, 50]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SASRec(args.hidden_factor, item_num, seq_size, args.dropout_rate, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    bce_loss = nn.BCEWithLogitsLoss()

    model.to(device)

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    ps = calcu_propensity_score(train_data)
    ps = torch.tensor(ps)
    ps = ps.to(device)

    total_step=0
    hr_max = 0
    best_epoch = 0

    aug_rate = args.aug_rate
    cl_rate = args.cl_rate
    
    num_rows=train_data.shape[0]
    num_batches=int(num_rows/args.batch_size)

    for i in range(args.epoch):
        for j in range(num_batches):
            batch = train_data.sample(n=args.batch_size).to_dict()
            seq = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target=list(batch['next'].values())

            target_neg = []
            for index in range(args.batch_size):
                neg=np.random.randint(item_num)
                while neg==target[index]:
                    neg = np.random.randint(item_num)
                target_neg.append(neg)
            optimizer.zero_grad()

            seq = torch.LongTensor(seq)
            seq_np = seq
            
            len_seq = torch.LongTensor(len_seq)
            target = torch.LongTensor(target)
            target_neg = torch.LongTensor(target_neg)
            seq = seq.to(device)
            target = target.to(device)
            len_seq = len_seq.to(device)
            target_neg = target_neg.to(device)

            model_output = model.forward(seq, len_seq)

            target = target.view(args.batch_size, 1)
            target_neg = target_neg.view(args.batch_size, 1)

            pos_scores = torch.gather(model_output, 1, target)
            neg_scores = torch.gather(model_output, 1, target_neg)

            pos_labels = torch.ones((args.batch_size, 1))
            neg_labels = torch.zeros((args.batch_size, 1))

            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((pos_labels, neg_labels), 0)
            labels = labels.to(device)

            rec_loss = bce_loss(scores, labels)
            '''

            # cl loss 1. 要不要加另外的数据增强方法
            aug_seq1, aug_pos1, aug_len1 = SequenceAugmentor.item_crop(seq_np, len_seq, aug_rate)
            aug_seq2, aug_pos2, aug_len2 = SequenceAugmentor.item_crop(seq_np, len_seq, aug_rate)
            aug_emb1 = model.forward(torch.LongTensor(aug_seq1).to(device), len_seq)
            aug_emb2 = model.forward(torch.LongTensor(aug_seq2).to(device), len_seq)
            
            '''
            
            # 定义一个数据增强的aug_type
            if args.aug_type == 0:
                # 使用裁剪方法生成第一个增强数据视图
                aug_seq1, aug_pos1, aug_len1 = SequenceAugmentor.item_crop(seq_np, len_seq, aug_rate)
                aug_seq2, aug_pos2, aug_len2 = SequenceAugmentor.item_crop(seq_np, len_seq, aug_rate)
                aug_emb1 = model.forward(torch.LongTensor(aug_seq1).to(device), len_seq)
                aug_emb2 = model.forward(torch.LongTensor(aug_seq2).to(device), len_seq)
                # 计算第一和第二个视图之间的对比损失
                cl_loss = cl_rate * InfoNCE(aug_emb1, aug_emb2, 1, True)
            elif args.aug_type == 1:
                # 使用重排序方法生成第二个增强数据视图
                aug_seq1 = SequenceAugmentor.item_reorder(seq_np, len_seq, aug_rate)
                aug_seq2 = SequenceAugmentor.item_reorder(seq_np, len_seq, aug_rate)
                aug_emb1 = model.forward(torch.LongTensor(aug_seq1).to(device), len_seq)
                aug_emb2 = model.forward(torch.LongTensor(aug_seq2).to(device), len_seq)
                # 计算第一和第三个视图之间的对比损失
                cl_loss = cl_rate * InfoNCE(aug_emb1, aug_emb2, 1, True)
            else:
                # 使用掩码方法生成第三个增强数据视图
                aug_seq1 = SequenceAugmentor.item_mask(seq_np, len_seq, aug_rate, mask_idx=0) # 假设掩码索引为0
                aug_seq2 = SequenceAugmentor.item_mask(seq_np, len_seq, aug_rate, mask_idx=0) # 假设掩码索引为0
                aug_emb1 = model.forward(torch.LongTensor(aug_seq1).to(device), len_seq)
                aug_emb2 = model.forward(torch.LongTensor(aug_seq2).to(device), len_seq)
                # 计算第二和第三个视图之间的对比损失
                cl_loss = cl_rate * InfoNCE(aug_emb1, aug_emb2, 1, True)
                

            loss = rec_loss + cl_loss

            pos_scores_dro = torch.gather(torch.mul(model_output * model_output, ps), 1, target)
            pos_scores_dro = torch.squeeze(pos_scores_dro)
            pos_loss_dro = torch.gather(torch.mul((model_output - 1) * (model_output - 1), ps), 1, target)
            pos_loss_dro = torch.squeeze(pos_loss_dro)

            inner_dro = torch.sum(torch.exp((torch.mul(model_output * model_output, ps) / args.beta)), 1) - torch.exp((pos_scores_dro / args.beta)) + torch.exp((pos_loss_dro / args.beta)) 

            loss_dro = torch.log(inner_dro + 1e-24)
            if args.alpha == 0.0:
                loss_all = loss
            else:
                loss_all = loss + args.alpha * torch.mean(loss_dro)
            loss_all.backward()
            optimizer.step()

            total_step += 1
            if total_step % 200 == 0:
                print("the loss in %dth step is: %f" % (total_step, loss_all))

            if total_step % 2000 == 0:
                    print('VAL PHRASE:')
                    hr_20 = evaluate(model, 'val_sessions.df', device)
                    print('TEST PHRASE:')
                    _ = evaluate(model, 'test_sessions.df', device)

                    if hr_20 > hr_max:
                        hr_max = hr_20
                        best_epoch = total_step
                    
                    print('BEST EPOCH:{}'.format(best_epoch))




                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

