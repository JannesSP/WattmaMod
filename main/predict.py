import argparse
import gc
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, get_worker_info, IterableDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.model import WaveCrossMamba, AnomalyDetectionModel
from utils.zstd import read_zstd, write_zstd

print("Project root:", os.getcwd(), file=sys.stderr)
kmer_encode_dic={'A': 0, "C": 1, "G": 2, "T": 3}

class PredictIterableDataset(IterableDataset):
    def __init__(self, file_path):
        super().__init__()
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
            print("Parse error:", e, file=sys.stderr)
            print(line, file=sys.stderr)
            raise


    def __iter__(self):
        info = get_worker_info()
        if info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = info.id, info.num_workers

        f, raw = read_zstd(self.file_path)
        try:
            # skip first line, which is the header
            next(f)
            for i, line in enumerate(f):
                if i % num_workers != worker_id:
                    continue

                result = self.parse_line(line)
                if result is not None:
                    yield result
        finally:
            f.close()
            if raw is not None:
                raw.close()

def predict_model(pretrain_model, model, test_loader, device, output_predict, max_write=None):
    predict_result, raw = write_zstd(output_predict)

    try:
        model.to(device)
        pretrain_model.to(device)
        model.eval()
        pretrain_model.eval()

        label_dict = {0: "unmod", 1: "mod"}
        written = 0

        with torch.no_grad():

            print("Starting prediction...", file=sys.stderr)

            pbar = tqdm(
                total=max_write,
                unit="reads",
                desc="Predicting",
                file=sys.stderr,
                mininterval=30,       # update at most every 30 seconds
                dynamic_ncols=False,  # better for Slurm log files
                leave=True
            )
            
            for batch_idx, (data, batch_y) in enumerate(test_loader):
                # print("Got batch", batch_idx, file=sys.stderr)
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

                    print(
                        f"{contig}\t{position}\t{motif}\t{read_id}\t{label_dict[int(pred[j])]}\t{float(probabilities[j])}",
                        file=predict_result
                    )

                written += take
                pbar.update(take)
            
    finally:

        pbar.close()
        predict_result.close()
        if raw is not None:
            stream, fh = raw
            stream.close()
            fh.close()
        
    print(f"Prediction completed. Total reads written: {written}", file=sys.stderr)

    return written

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = PredictIterableDataset(args.input)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=10_000, 
        num_workers=0,
        pin_memory=True
    )

    pretrained_feature_extractor  = WaveCrossMamba(device=device,d_model=128).to(device)
    fine_tune_model  = AnomalyDetectionModel(feature_dim=128, num_classes=2).to(device)
    pretrained_feature_extractor.load_state_dict(torch.load(args.pre,map_location=device))
    fine_tune_model.load_state_dict(torch.load(args.fine, map_location=device))

    n_written = predict_model(pretrained_feature_extractor, fine_tune_model, test_loader, device, args.output, args.max_n)
    del pretrained_feature_extractor, fine_tune_model, test_loader, test_dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='WattmaMod: multi-type RNA modification prediction.')
    parser.add_argument('-p', '--pre', type=str, required=True, help='Path to the pretrained encoder checkpoint (.pth).')
    parser.add_argument('-f', '--fine', type=str, required=True, help='Path to the fine-tuned classifier checkpoint (.pth).')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file (TSV), will be zst compressed (lvl 3).')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input TSV file for prediction.')
    parser.add_argument('-g', '--gpu', type=str, default="0", help='GPU device id(s) to use, e.g., "0", "1".')
    parser.add_argument('-m', '--max_n', type=int, default=None, help='Maximum number of reads to predict (default: all).')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)

