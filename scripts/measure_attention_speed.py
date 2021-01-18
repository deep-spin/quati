# profiling:
# python3 -m cProfile -o attns.prof measure_attention_speed.py
# snakeviz attns.prof
import argparse
import cProfile
import time

import torch

from quati.modules.attention import Attention
from quati.modules.continuous_attention_seq2seq import ContinuousAttention
from quati.modules.continuous_encoders import ContinuousEncoder
from quati.modules.multi_headed_attention import MultiHeadedAttention
from quati.modules.scorer import DotProductScorer, GeneralScorer


GPU_ID = None


class AverageEncoder(ContinuousEncoder):
    """
    This encoder is only used for performance experiments.
    """
    def __init__(self, vector_size, kind='linear'):
        super().__init__()
        self.W = torch.nn.Parameter(
            torch.randn(2, vector_size, vector_size // 2),
            requires_grad=True
        )
        self.conv = None
        if kind == 'conv':
            self.conv = torch.nn.Conv1d(vector_size,
                                        vector_size,
                                        kernel_size=3)

    def forward(self, query, keys, mask=None):
        if self.conv is not None:
            x = self.conv(keys.transpose(-1, -2)).transpose(-1, -2)
            pool = x.max(dim=1)[0]
        else:
            pool = torch.mean(keys, dim=1)
        zk = pool.matmul(self.W)
        zq = query.matmul(self.W)
        x = zq.matmul(zk.transpose(-1, -2))
        x = x.transpose(0, -1)
        mu = torch.sigmoid(x[:, :, 0])
        sigma_sq = torch.nn.functional.softplus(x[:, :, 1])
        return mu, sigma_sq, None


def timeit(mod, *args, **kwargs):
    global GPU_ID
    n = 10
    t = 0
    result = None
    for _ in range(n):
        if torch.cuda.is_available() and GPU_ID is not None:
            mod = mod.cuda()
            torch.cuda.empty_cache()  # clear cache before timing
            torch.cuda.synchronize(GPU_ID)  # wait for initialization to finish
        time1 = time.perf_counter()
        result = mod(*args, **kwargs)
        if torch.cuda.is_available() and GPU_ID is not None:
            torch.cuda.synchronize(GPU_ID)
        time2 = time.perf_counter()
        t += (time2 - time1)
    print('Elapsed time: {}'.format(t / n))
    return t / n, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Time performance")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--nb-heads", type=int, default=4)
    parser.add_argument("--nb-waves", type=int, default=8)
    parser.add_argument("--wave", action="store_true")
    parser.add_argument("--gaussian", action="store_true")
    parser.add_argument("--power", action="store_true")
    parser.add_argument("--multihead", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--filename", type=str, default='times.txt')
    args = parser.parse_args()

    batch_size = args.batch_size
    source_len = args.seq_len
    target_len = args.seq_len
    vector_size = args.vector_size
    nb_heads = args.nb_heads
    nb_waves = args.nb_waves
    seq_lens = [50 * i for i in range(1, 50)]
    max_seq_len = seq_lens[-1]
    GPU_ID = args.gpu_id

    with open(args.filename, 'w', encoding='utf8') as report:

        names = ['disc. dotp.', 'disc. general',
                 'cont. avg', 'cont. conv',
                 'mh disc. dotp.', 'mh disc. general',
                 'mh cont. avg', 'mh cont. conv',]

        # filter multihead attention from names in case args.multihead is False
        names = [name for name in names if args.multihead is True or (
                args.multihead is False and not name.startswith('mh'))]

        for name in names[:-1]:
            report.write('{},'.format(name))
        report.write('{}\n'.format(names[-1]))

        disc_attn = Attention(
            scorer=None,
            max_activation='sparsemax'
        )
        cont_attn = ContinuousAttention(
            encoder=None,
            nb_waves=nb_waves,
            max_seq_len=max_seq_len,
            use_power_basis=args.power,
            use_gaussian_basis=args.gaussian,
            use_wave_basis=args.wave,
            dynamic_nb_basis=False,
            consider_pad=False,
            max_activation='sparsemax',
            gpu_id=args.gpu_id,
            seq_lens=seq_lens
        )
        multihead_attn = MultiHeadedAttention(
            attn=None,
            nb_heads=nb_heads,
            query_size=vector_size,
            key_size=vector_size,
            value_size=vector_size,
            hidden_size=vector_size * nb_heads
        )

        for seq_len in seq_lens:
            source_len = target_len = seq_len

            keys = torch.randn(batch_size, source_len, vector_size, device=args.gpu_id)
            query = torch.randn(batch_size, target_len, vector_size, device=args.gpu_id)
            values = keys
            mask = torch.ones(batch_size, source_len, device=args.gpu_id)

            #
            # Discrete attention:
            #
            print('Discrete Attn - DotProduct Scorer:')
            try:
                disc_attn.scorer = DotProductScorer()
                el, (context_vector, probas) = timeit(disc_attn, query, keys, values, mask=mask)
                print(context_vector.shape, probas.shape)
            except RuntimeError as e:  # out of memory error
                el = float('inf')
            report.write('{},'.format(el))

            print('Discrete Attn - General Scorer:')
            try:
                disc_attn.scorer = GeneralScorer(vector_size, vector_size)
                el, (context_vector, probas) = timeit(disc_attn, query, keys, values, mask=mask)
                print(context_vector.shape, probas.shape)
            except RuntimeError as e:  # out of memory error
                el = float('inf')
            report.write('{},'.format(el))
            # cProfile.run('attn(query, keys, values, mask)',
            #              'd_attn_general_{}.prof'.format(seq_len))

            #
            # Continuous attention:
            #
            print('Continuous Attn - Avg Encoder:')
            cont_attn.encoder = AverageEncoder(vector_size, kind='linear')
            el, (context_vector, r) = timeit(cont_attn, query, keys, values, mask=mask)
            print(context_vector.shape, r.shape)
            report.write('{},'.format(el))
            # cProfile.run('attn(query, keys, values, mask)',
            #              'c_attn_avg_{}.prof'.format(seq_len))

            print('Continuous Attn - Conv Encoder:')
            cont_attn.encoder = AverageEncoder(vector_size, kind='conv')
            el, (context_vector, r) = timeit(cont_attn, query, keys, values,
                                             mask=mask)
            print(context_vector.shape, r.shape)
            report.write('{}'.format(el))

            if not args.multihead:
                report.write('\n')
            else:
                report.write(',')

            if args.multihead:
                #
                # Multihead Discrete attention
                #
                print('Discrete MultiHeadAttn - DotProduct Scorer:')
                disc_attn.scorer = DotProductScorer()
                multihead_attn.attention = disc_attn
                el, (context_vector, probas) = timeit(multihead_attn, query, keys, values, mask=mask)
                print(context_vector.shape, probas.shape)
                report.write('{},'.format(el))

                print('Discrete MultiHeadAttn - General Scorer:')
                disc_attn.scorer = GeneralScorer(vector_size, vector_size)
                multihead_attn.attention = disc_attn
                el, (context_vector, probas) = timeit(multihead_attn, query, keys, values, mask=mask)
                print(context_vector.shape, probas.shape)
                report.write('{},'.format(el))

                #
                # Multihead Continuous attention:
                #
                print('Continuous MultiHeadAttn - Avg Encoder:')
                cont_attn.encoder = AverageEncoder(vector_size, kind='linear')
                multihead_attn.attention = cont_attn
                el, (context_vector, r) = timeit(multihead_attn, query, keys, values, mask=mask)
                print(context_vector.shape, r.shape)
                report.write('{},'.format(el))

                print('Continuous MultiHeadAttn - Conv Encoder:')
                cont_attn.encoder = AverageEncoder(vector_size, kind='conv')
                multihead_attn.attention = cont_attn
                el, (context_vector, r) = timeit(multihead_attn, query, keys, values, mask=mask)
                print(context_vector.shape, r.shape)
                report.write('{},'.format(el))

            del keys, query, values, mask
            torch.cuda.empty_cache()

