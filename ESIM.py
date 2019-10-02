from torch import nn
import torch
import torch.nn.functional as F


class ESIM(nn.Module):
    def __init__(self, args):
        super(ESIM, self).__init__()
        self.args = args
        self.dropout = 0.5
        self.hidden_size = args.hidden_size
        self.embeds_dim = args.embeds_dim
        num_word = 20000
        self.embeds = nn.Embedding(num_word, self.embeds_dim)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size * 8, self.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size, args.linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(args.linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(args.linear_size, 2),
            nn.Softmax(dim=-1)
        )

    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        关于mask 原来是这样
                tensor([[1, 0, 0],
                [0, 1, 0],
                [0, 1, 0]], dtype=torch.uint8)
                fill完inf后变成这样
        tensor([[-inf, 0., 0.],
                [0., -inf, 0.],
                [0., -inf, 0.]])
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))

####### x1  x2  batchsize  *seq_len *embedding_size


        #mask和x1 的维度是一样的   batch_size *seq_len * embedding_size
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)


        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, *input):
        # batch_size * seq_len
        sent1, sent2 = input[0], input[1]
        mask1, mask2 = sent1.eq(0), sent2.eq(0)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity















    ###########修改后

    class ESIM(nn.Module):
        def __init__(self, args):
            super(ESIM, self).__init__()
            self.args = args
            self.dropout = 0.5
            self.hidden_size = args.hidden_size
            self.embeds_dim = args.embeds_dim
            self.pad_idx = args.pad_idx
            self.num_word = args.num_word

            self.embeds = nn.Embedding(self.num_word, self.embeds_dim, padding_idx=self.pad_idx)
            self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
            self.lstm1 = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(self.hidden_size * 8, self.hidden_size, batch_first=True, bidirectional=True)

            self.fc = nn.Sequential(
                nn.BatchNorm1d(self.hidden_size * 8),
                nn.Linear(self.hidden_size * 8, args.linear_size),
                nn.ELU(inplace=True),
                # nn.BatchNorm1d(args.linear_size),
                # nn.Dropout(self.dropout),
                # nn.Linear(args.linear_size, args.linear_size),
                # nn.ELU(inplace=True),
                nn.BatchNorm1d(args.linear_size),
                nn.Dropout(self.dropout),
                nn.Linear(args.linear_size, 4)
                # nn.Softmax(dim=-1)
            )

        def soft_attention_align(self, x1, x2, mask1, mask2):
            '''
            x1: batch_size * seq_len * dim
            x2: batch_size * seq_len * dim
            '''
            # attention: batch_size * seq_len * seq_len
            attention = torch.matmul(x1, x2.transpose(1, 2))
            mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
            mask2 = mask2.float().masked_fill_(mask2, float(
                '-inf'))  ###为什么叫mask，就像面具一样，把原来<pad>的值全部用 -inf覆盖了，这样softmax出来就都是0 ，也就是attention中和<pad>相关的注意力都是0,计算attention中常用的技巧
            # mask       batch_size  *seq_len    unsqueeze之后变成batch_size *1 *seq_len

            # weight: batch_size * seq_len * seq_len

            weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)  # batch_size *(seq_len)*seq_len 注意矩阵维度不一样的时候的加法
            # '''[2,3,3]+[2,1,3]
            # torch.Size([2, 3, 3])
            # tensor([[[-0.3740, 1.0252, 0.5291],
            #          [-0.2367, 2.1526, -0.2691],
            #          [-1.6116, -1.5094, 0.2614]],
            #
            #         [[-0.8805, 1.8062, 0.2487],
            #          [-0.0205, 0.1859, -0.5070],
            #          [0.2409, 0.5222, 1.0360]]])
            # tensor([[[1.3760, 0.6039, 0.2657]],
            #
            #         [[0.6044, 0.7289, -0.6047]]])
            # tensor([[[1.0020, 1.6291, 0.7947],
            #          [1.1394, 2.7565, -0.0035],
            #          [-0.2356, -0.9055, 0.5270]],
            #
            #         [[-0.2760, 2.5351, -0.3560],
            #          [0.5839, 0.9148, -1.1117],
            #          [0.8453, 1.2511, 0.4312]]])
            # '''
            x1_align = torch.matmul(weight1, x2)
            weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
            x2_align = torch.matmul(weight2,
                                    x1)  # batch_size*seq_len *seq_len         batch_size * seq_len * hidden_size
            # x_align: batch_size * seq_len * hidden_size
            return x1_align, x2_align

        def submul(self, x1, x2):
            mul = x1 * x2  ###############element-wise 相应元素相乘,论文中是这么写的吗???? 好像是的  arxiv 1609.06038v3  Enhanced Lstm for natural language Inference 第
            sub = x1 - x2
            return torch.cat([sub, mul], -1)  ####按最后一个维度进行拼接

        def apply_multiple(self, x):
            # input: batch_size * seq_len * (2 * hidden_size)
            p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
            p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
            # output: batch_size * (4 * hidden_size)
            return torch.cat([p1, p2], 1)

        def forward(self, sentence1, sentence2):
            # batch_size * seq_len
            sent1, sent2 = sentence1, sentence2

            mask1, mask2 = sent1.eq(0), sent2.eq(0)  ####这里的mask其实就是原始句子中的<pad>标签把

            # embeds: batch_size * seq_len => batch_size * seq_len * dim
            x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1,
                                                                                           2)  # batch_size *seq_len * 2*embedding_dim
            x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

            # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
            o1, _ = self.lstm1(x1)
            o2, _ = self.lstm1(x2)

            # Attention
            # batch_size * seq_len * hidden_size
            q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

            # Compose
            # batch_size * seq_len * (8 * hidden_size)
            q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
            q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

            # batch_size * seq_len * (2 * hidden_size)
            q1_compose, _ = self.lstm2(q1_combined)
            q2_compose, _ = self.lstm2(q2_combined)

            # Aggregate
            # input: batch_size * seq_len * (2 * hidden_size)
            # output: batch_size * (4 * hidden_size)
            q1_rep = self.apply_multiple(q1_compose)
            q2_rep = self.apply_multiple(q2_compose)

            # Classifier
            x = torch.cat([q1_rep, q2_rep], -1)
            similarity = self.fc(x)

            return similarity

##################初始化需要哪些东西??? hidden_size   embeds_dim     linear_size

    INPUT_DIM1 = len(SENTENCE1.vocab)
    INPUT_DIM2 = len(SENTENCE2.vocab)
    EMBEDS_DIM = 100
    HIDDEN_SIZE = 128
    LINEAR_SIZE = 50
    OUTPUT_DIM = 3
    DROPOUT = 0.5

    PAD_IDX = SENTENCE1.vocab.stoi[SENTENCE1.pad_token]

    # PAD_IDX2 =SENTENCE2.vocab.stoi[SENTENCE1.pad_token]
    #
    # print("pad_idx1:",PAD_IDX1)
    #
    # print("pad_idx2:",PAD_IDX2)

    class arg:
        def __init__(self, a, b, c, d, e):
            self.hidden_size = a
            self.embeds_dim = b
            self.linear_size = c
            self.pad_idx = d
            self.num_word = e

    NUM_WORD = len(SENTENCE1.vocab)
    ARG = arg(HIDDEN_SIZE, EMBEDS_DIM, LINEAR_SIZE, PAD_IDX, NUM_WORD)

    model = ESIM(ARG)
