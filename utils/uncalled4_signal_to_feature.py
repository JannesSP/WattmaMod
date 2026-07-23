import argparse
import traceback
import numpy as np
import re
import sys
from pathlib import Path
from statsmodels import robust
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.zstd import read_zstd, write_zstd

# def motif_center_indices(sequence: str, motif_regex) -> list[int]:
#     n = len(sequence)
#     idx = []
#     if n < 5:
#         return idx
#     for i in range(2, n-2):
#         kmer = sequence[i-2:i+3]
#         if motif_regex is None or motif_regex.fullmatch(kmer):
#             idx.append(i)
#     return idx

def interp(x):
    x = np.asarray(x, dtype=np.float32)
    l = len(x)

    if l == 0:
        return []

    if l == 1:
        return [round(float(x[0]), 4)] * 100

    xp = np.linspace(0, 1, l)
    xnew = np.linspace(0, 1, 100)

    return np.round(np.interp(xnew, xp, x), 4).tolist()

def convert_base_name(base_name):
    merge_bases = {
        'A': 'A',
        'C': 'C',
        'G': 'G',
        'T': 'T',
        'M': '[AC]',
        'V': '[ACG]',
        'R': '[AG]',
        'H': '[ACT]',
        'W': '[AT]',
        'D': '[AGT]',
        'S': '[CG]',
        'B': '[CGT]',
        'Y': '[CT]',
        'N': '[ACGT]',
        'K': '[GT]'
    }
    pattern = ''
    for base in base_name:
        pattern += merge_bases.get(base, base)
    return pattern

def extract_5mer_features(signal_file: str, args):
    motif_regex = re.compile(convert_base_name(args.motif)) if args.motif else None
    out, raw_out = write_zstd(args.out)
    header = [
        "read_id","chr","pos1","5mer",
        "mean","std","median","length","base_qual",
        "sig_-2","sig_-1","sig_0","sig_+1","sig_+2"
    ]
    out.write("\t".join(header) + "\n")

    count = 0
    out_buffer = []
    buffer_size = 10000   # number of output lines before writing
    f, raw = read_zstd(signal_file)
        
    try:
        for line in tqdm(f, desc="Processing reads", unit=" reads", file=sys.stderr, mininterval=30):
            try:
                line = line.rstrip("\n")
                if not line or line.startswith("read_id"):
                    continue
                items = line.split("\t")
                read_id = items[0]
                chrom   = items[1]
                start   = int(items[2])     
                ref_seq = items[3]
                base_q = items[4]
                sequence = items[5]        
                dwell = items[6]
                sig_str   = items[7]
                base_quality_list = [int(x) for x in base_q.split("|") if x != ""]
                dwell_list = [int(x) for x in dwell.split(",") if x != ""]

                per_base_segments = []
                for part in sig_str.split("|"):
                    seg = np.fromstring(part, sep=",", dtype=np.float32)
                    seg = seg[np.isfinite(seg)]
                    per_base_segments.append(seg)
                full_signal = np.concatenate(per_base_segments)

                if full_signal.size == 0:
                    continue
                full_signal_uniq = np.unique(full_signal)
                med = float(np.median(full_signal_uniq))
                mad = float(robust.mad(full_signal_uniq))

                non_gap_prefix = [0] * (len(ref_seq) + 1)
                for idx, b in enumerate(ref_seq):
                    non_gap_prefix[idx+1] = non_gap_prefix[idx] + (1 if b != "-" else 0)

                for i in range(10, len(ref_seq) - 18):
                    if ref_seq[i] == "-":
                        continue
                    
                    kmer_sequence = sequence[i-2:i+3]
                    if motif_regex and not motif_regex.fullmatch(kmer_sequence):
                        continue
                    if kmer_sequence != ref_seq[i-2:i+3]:
                        continue
                    if i+2 >= len(per_base_segments):
                        break

                    five_segments = [
                        per_base_segments[i-2],
                        per_base_segments[i-1],
                        per_base_segments[i],
                        per_base_segments[i+1],
                        per_base_segments[i+2],
                    ]
                    
                    five_norm = [(s - med) / mad for s in five_segments]

                    means = []
                    stds = []
                    medians = []
                    five_interp = []
                    for s in five_norm:
                        means.append(float(np.round(np.mean(s), 3)))
                        stds.append(float(np.round(np.std(s), 3)))
                        medians.append(float(np.round(np.median(s), 3)))
                        five_interp.append(interp(s))

                    length5 = dwell_list[i-2:i+3]
                    baseq5  = base_quality_list[i-2:i+3]
                    if len(length5) != 5 or len(baseq5) != 5:
                        continue
                    offset = non_gap_prefix[i]
                    pos1 = start + offset

                    out_buffer.append(
                        "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                            read_id, chrom, pos1, kmer_sequence,
                            "|".join(map(str, means)),
                            "|".join(map(str, stds)),
                            "|".join(map(str, medians)),
                            "|".join(map(str, length5)),
                            "|".join(map(str, baseq5)),
                            "|".join(map(str, five_interp[0])),
                            "|".join(map(str, five_interp[1])),
                            "|".join(map(str, five_interp[2])),
                            "|".join(map(str, five_interp[3])),
                            "|".join(map(str, five_interp[4])),
                        )
                    )

                    if len(out_buffer) >= buffer_size:
                        out.write("".join(out_buffer))
                        out_buffer.clear()
                    
                    count += 1

            except Exception as e:
                print(e, file=sys.stderr)
                traceback.print_exc()

    finally:
        f.close()
        if raw is not None:
            raw.close()

        if out_buffer:
            out.write("".join(out_buffer))
            out_buffer.clear()
            
        out.close()
        if raw_out is not None:
            stream, fh = raw_out
            stream.close()
            fh.close()
    
    print(f"[*] done. total {count} 5-mer features → {args.out}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Extract 5-mer signal features based on sequence consistency and motif scanning.")
    ap.add_argument("-s", "--signal", required=True, type=str, help="Input signal file (.zst-compressed)")
    ap.add_argument("-m", "--motif", required=True, type=str, help="5-mer IUPAC pattern (exact match)")
    ap.add_argument("-o", "--out", required=True, type=str, help="Output feature file (.zst-compressed)")
    args = ap.parse_args()
    extract_5mer_features(args.signal, args)

if __name__ == "__main__":
    main()