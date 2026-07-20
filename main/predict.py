import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info, IterableDataset
from pathlib import Path
CUR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CUR_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print("Project root:", os.getcwd())
from model.model import WaveCrossMamba, AnomalyDetectionModel
kmer_encode_dic={'A': 0, "C": 1, "G": 2, "T": 3}

class PredictIterableDataset(IterableDataset):
    def __init__(self, file_path):
        super(PredictIterableDataset).__init__()
        self.file_path = file_path
    def parse_line(self, line):
        items = line.strip().split("\t")
        if len(items) < 14:
            return None
        try:
            read_id = items[0]
            contig = items[1]
            position = items[2]
            motif = items[3]

            signal = np.array([float(x) for x in "|".join(items[9:14]).split("|")])
            kmer = np.array([kmer_encode_dic[base] for base in motif])
            mean = np.array([float(x) for x in items[4].split("|")])
            std = np.array([float(x) for x in items[5].split("|")])
            intense = np.array([float(x) for x in items[6].split("|")])
            dwell = np.array([float(x) for x in items[7].split("|")]) / 200.0
            base_quality = np.array([float(x) for x in items[8].split("|")]) / 40.0
            x = [
                torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(2),
                torch.tensor(kmer, dtype=torch.long),
                torch.tensor(mean, dtype=torch.float32),
                torch.tensor(std, dtype=torch.float32),
                torch.tensor(intense, dtype=torch.float32),
                torch.tensor(dwell, dtype=torch.float32),
                torch.tensor(base_quality, dtype=torch.float32),
            ]
            y = "|".join([contig, position, motif, read_id])
            return x, y
        except Exception as e:
            return None


    def __iter__(self):
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        with open(self.file_path, "r") as f:
            for i, line in enumerate(f):
                if i % num_workers != worker_id:
                    continue
                result = self.parse_line(line)
                if result is not None:
                    yield result

def predict_model(pretrain_model, model, test_loader, device, output_predict, max_write=None):
    with open(output_predict, "w") as predict_result:
        model.to(device)
        pretrain_model.to(device)
        model.eval()
        pretrain_model.eval()

        label_dict = {0: "unmod", 1: "mod"}
        written = 0

        with torch.no_grad():
            for batch_idx, (data, batch_y) in enumerate(test_loader):
                if max_write is not None and written >= max_write:
                    break

                x, kmer, mean, std, intense, dwell, base_quality = data
                x = x.to(device)
                kmer = kmer.to(device)
                mean = mean.to(device)
                std = std.to(device)
                intense = intense.to(device)
                dwell = dwell.to(device)
                base_quality = base_quality.to(device)

                logits, ff = pretrain_model(x, kmer, mean, std, intense, dwell, base_quality)
                out = model(logits)

                out = torch.softmax(out, dim=1)
                pred = torch.max(out, 1)[1].cpu().numpy()
                probabilities = out.detach().cpu().numpy()[:, 1]

                bs = len(batch_y)
                take = bs if max_write is None else min(bs, max_write - written)

                for j in range(take):
                    label_str = batch_y[j]
                    parts = label_str.split("|")
                    position, motif, read_id = parts[-3:]
                    contig_full = "|".join(parts[:-3])
                    contig = contig_full.split("|", 1)[0]

                    print("%s\t%s\t%s\t%s\t%s\t%s" % (
                        contig,
                        position,
                        motif,
                        read_id,
                        label_dict[int(pred[j])],
                        float(probabilities[j])
                    ), file=predict_result)

                written += take
                if (written % 1_000_000) < take:
                    print(f"[INFO]  {written} ")

        return written

import gc, torch

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = PredictIterableDataset(args.input)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=10000, 
        num_workers=4,
        pin_memory=True
    )

    pretrained_feature_extractor  = WaveCrossMamba(device=device,d_model=128).to(device)
    fine_tune_model  = AnomalyDetectionModel(feature_dim=128, num_classes=2).to(device)
    pretrained_feature_extractor.load_state_dict(torch.load(args.pre,map_location=device))
    fine_tune_model.load_state_dict(torch.load(args.fine, map_location=device))
    max_n = getattr(args, "max_n", 8_000_000)

    n_written = predict_model(pretrained_feature_extractor, fine_tune_model, test_loader, device, args.o, max_write=max_n)
    del pretrained_feature_extractor, fine_tune_model, test_loader, test_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='WattmaMod: multi-type RNA modification prediction.')
    parser.add_argument('--pre', type=str, required=True,help='Path to the pretrained encoder checkpoint (.pth).')
    parser.add_argument('--fine', type=str, required=True,help='Path to the fine-tuned classifier checkpoint (.pth).')
    parser.add_argument('--o', '-output_predict', type=str, required=True,help='Output file (TSV).')
    parser.add_argument('--input', '--input_data_file', type=str, required=True,help='Input TSV file for prediction.')
    parser.add_argument('--gpu', type=str, default="0",help='GPU device id(s) to use, e.g., "0", "1".')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)

