from fastNLP import DataSet, Instance
from fastNLP.io import Loader


class DataLoader(Loader):
    def __init__(self, max_seq_len=150):
        super().__init__()
        self.max_seq_len=max_seq_len

    def _load(self, path: str) -> DataSet:
        print('Loading {}...'.format(path))
        total_sampls, debias_samples, distillation_samples = 0, 0, 0
        ds = DataSet()
        with open(path, 'r') as fin:
            lines = fin.readlines()
            for l in lines:
                items = l.split('\t')
                refs = ' '.join(items[0].strip().split(' ')[:self.max_seq_len])
                hyps = ' '.join(items[1].strip().split(' ')[:self.max_seq_len])
                sample = {
                    'refs': refs,
                    'hyps': hyps,
                    'labels': float(items[2]),
                    'type': items[3],
                }
                ds.append(Instance(**sample))
                # statistics
                total_sampls += 1
                if sample['type'] == 'debias':
                    debias_samples += 1
                else:
                    distillation_samples += 1

        ds.set_input("refs", "hyps", "labels")
        ds.set_target("labels")
        return ds
