import argparse, sys, os
import pysam, pod5
import numpy as np
from array import array as pyarray
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm
from utils.zstd import write_zstd
REF_DICT = None
P5_READER = None
P5_PATH = None

def reverse_non_gap_segments(parts, dwell_padded, ref_aligned_sequence, na_token="NA"):
    n = len(parts)
    if n == 0:
        return parts, dwell_padded

    ref_prefix = ref_aligned_sequence[:n]
    gap_mask = [ (b == '-') for b in ref_prefix ]
    idx_non_gap = [i for i, is_gap in enumerate(gap_mask) if not is_gap]
    if not idx_non_gap:
        return parts, dwell_padded

    new_parts = list(parts)
    new_dwell = list(dwell_padded)

    for src, dst in zip(idx_non_gap, reversed(idx_non_gap)):
        seg   = parts[src]
        dwell = dwell_padded[src]

        if seg and seg not in ("0", na_token):
            vals = seg.split(",")
            vals.reverse() 
            new_parts[dst] = ",".join(vals)
        else:
            new_parts[dst] = seg

        new_dwell[dst] = dwell

    return new_parts, new_dwell


def init_worker(pod5_path, ref_dict):

    global REF_DICT, P5_READER, P5_PATH
    P5_PATH = pod5_path
    REF_DICT = ref_dict
    P5_READER = pod5.DatasetReader(P5_PATH)
def load_signal_dict(pod5_path):
    signal_dict = {}
    if os.path.isdir(pod5_path):
        files = [os.path.join(pod5_path, f) for f in os.listdir(pod5_path) if f.endswith(".pod5")]
    else:
        files = [pod5_path]
    for p in files:
        with pod5.Reader(p) as reader:
            for read in reader.reads():
                sig = getattr(read, "signal", None)
                if sig is None:
                    sig = getattr(read, "raw_signal", None)
                if sig is not None:
                    signal_dict[str(read.read_id)] = np.asarray(sig, dtype=np.int32)
    return signal_dict

def load_fasta_dict(reference: str) -> dict[str, str]:
    fa = pysam.FastaFile(reference)
    d = {}

    ref_names = list(fa.references)
    for name in tqdm(ref_names, desc="Loading FASTA", unit="seq"):
        d[name] = fa.fetch(name)

    fa.close()
    return d
def get_ur_start(ur_tag):
    vals = ur_tag.tolist() if isinstance(ur_tag, pyarray) else list(ur_tag)
    if len(vals) < 2 or len(vals) % 2 != 0:
        return None
    return vals[0]

def extract_aligned_sequences_for_ur_region(
    chrom,
    ref_start,
    cigar_tuples,
    read_seq,
    read_qual,
    ref_dict,
    ur_tag,
):
    if chrom not in ref_dict or read_seq is None:
        return None, None, None

    full_ref_seq = ref_dict[chrom]
    aligned_ref = []
    aligned_read = []
    aligned_qual = []

    read_pos = 0
    ref_pos = ref_start

    for op, length in cigar_tuples:
        if op in (0, 7, 8):  # M, =, X
            ref_segment = full_ref_seq[ref_pos:ref_pos + length]
            aligned_ref.extend(list(ref_segment))

            read_segment = read_seq[read_pos:read_pos + length]
            aligned_read.extend(list(read_segment))

            if read_qual:
                qual_segment = read_qual[read_pos:read_pos + length]
                aligned_qual.extend(qual_segment)
            else:
                aligned_qual.extend([0] * length)

            ref_pos += length
            read_pos += length

        elif op == 1:  # I
            aligned_ref.extend(['-'] * length)

            read_segment = read_seq[read_pos:read_pos + length]
            aligned_read.extend(list(read_segment))

            if read_qual:
                qual_segment = read_qual[read_pos:read_pos + length]
                aligned_qual.extend(qual_segment)
            else:
                aligned_qual.extend([0] * length)

            read_pos += length

        elif op == 2:  # D
            ref_segment = full_ref_seq[ref_pos:ref_pos + length]
            aligned_ref.extend(list(ref_segment))
            aligned_read.extend(['-'] * length)
            aligned_qual.extend([0] * length)
            ref_pos += length

        elif op == 4:  # soft-clip
            read_pos += length
        elif op == 5:  # hard-clip
            pass
        elif op == 3:  # N
            ref_pos += length

    ref_aligned_full = ''.join(aligned_ref)
    read_aligned_full = ''.join(aligned_read)
    qual_full = aligned_qual

    vals = ur_tag.tolist() if isinstance(ur_tag, pyarray) else list(ur_tag)
    if len(vals) < 2 or len(vals) % 2 != 0:
        raise ValueError("invalid ur_tag")
    ur_blocks = [(vals[k], vals[k + 1]) for k in range(0, len(vals), 2)]

    ref_pos_current = ref_start
    blk_idx = 0
    ur_ref_indices = []

    for i, ref_base in enumerate(ref_aligned_full):
        while blk_idx < len(ur_blocks) and ref_pos_current >= ur_blocks[blk_idx][1]:
            blk_idx += 1

        in_block = (
            blk_idx < len(ur_blocks)
            and ur_blocks[blk_idx][0] <= ref_pos_current < ur_blocks[blk_idx][1]
        )

        if ref_base == '-':
            if in_block:
                ur_ref_indices.append(i)
        else:
            if in_block:
                ur_ref_indices.append(i)
            ref_pos_current += 1

    ref_aligned_ur = ''.join(ref_aligned_full[i] for i in ur_ref_indices)
    read_aligned_ur = ''.join(read_aligned_full[i] for i in ur_ref_indices)
    qual_ur = [qual_full[i] for i in ur_ref_indices]
    qual_str = '|'.join(map(str, qual_ur))

    return ref_aligned_ur, read_aligned_ur, qual_str
REF_DICT = None
P5_READER = None


def iter_tasks(bam, ref_dict):
    for rec in bam.fetch(until_eof=True):
        if rec.is_unmapped or rec.is_secondary or rec.is_supplementary:
            continue
        if rec.is_reverse:
            continue
        chrom = rec.reference_name
        if chrom not in ref_dict:
            continue
        if not rec.has_tag("ur") or not rec.has_tag("ul"):
            continue

        read_id = rec.query_name.split()[0]
        ur_tag = rec.get_tag("ur")
        ul_tag = rec.get_tag("ul")

        yield (
            read_id,
            chrom,
            rec.reference_start,
            rec.cigartuples,
            rec.query_sequence,
            rec.query_qualities,
            ur_tag,
            ul_tag,
        )


def process_read(task):

    (
        read_id,
        chrom,
        ref_start,
        cigar_tuples,
        read_seq,
        read_qual,
        ur_tag,
        ul_tag,
    ) = task
    try:
        global REF_DICT, P5_READER

        pod5_read = P5_READER.get_read(read_id)
        sig = getattr(pod5_read, "signal", None)
        if sig is None:
            sig = getattr(pod5_read, "raw_signal", None)
        if sig is None:
            return None
        sig_arr = np.asarray(sig, dtype=np.int32)
        ref_aligned, read_aligned, qual_str = extract_aligned_sequences_for_ur_region(
            chrom,
            ref_start,
            cigar_tuples,
            read_seq,
            read_qual,
            REF_DICT,
            ur_tag,
        )
        if ref_aligned is None:
            return None

        signal, dwell_str = extract_signal(ul_tag, sig_arr, ref_aligned)
        ur_start = get_ur_start(ur_tag)
        if ur_start is None:
            return None
        start_pos_1based = ur_start + 1

        line = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
            read_id,
            chrom,
            start_pos_1based,
            ref_aligned,
            qual_str,
            read_aligned,
            dwell_str,
            signal,
        )
        return line

    except Exception as e:
        print(f"Error processing read {read_id}: {e}", file=sys.stderr)
        return None


def extract_reference_from_ur(ref_dict, chrom, ur_tag):
    if chrom not in ref_dict:
        return None
    ref_sequence_parts = []
    
    for i in range(0, len(ur_tag), 2):
        start = ur_tag[i]
        end = ur_tag[i + 1]
        ref_sequence_parts.append(ref_dict[chrom][start:end])
    return "".join(ref_sequence_parts)

def extract_signal(ul, sig_win, ref_aligned_sequence, na_token="NA"):

    ul = np.asarray(ul, dtype=np.int64)
    
    sig = np.asarray(sig_win)
    N = int(len(sig))
    p = 0
    i = 0 
    
    while i < len(ul) and ul[i] < 0:
        p += int(-ul[i])
        i += 1

    lens_all = ul[i:]
    lens = []
    intervals = []
    
    for v in map(int,lens_all):
        if v < 0:
            p += -v
            continue
        if v ==0:
            intervals.append(None)
            lens.append(0)
        else:
            s,e = p, p + v
            intervals.append((s,e))
            lens.append(v)
            p = e
    n_pos = len(intervals)
    last_seg = None
    
    for j in range(n_pos):
        if intervals[j] is not None:
            last_seg = intervals[j]
        else:
            if last_seg is not None:
                intervals[j] = last_seg

    if n_pos > 0 and intervals[0] is None:
        nxt = None
        for j in range(n_pos - 1, -1, -1):
            if intervals[j] is not None:
                nxt = intervals[j]
            else:
                if nxt is not None:
                    intervals[j] = nxt

    parts = []
    empty_idx = []
    dwell_padded = []

    lens_index = 0
    for idx, base in enumerate(ref_aligned_sequence):
        if base == '-':
            parts.append("0")
            dwell_padded.append(0)
            continue

        if lens_index >= len(intervals):
            break

        seg = intervals[lens_index]
        
        v_dwell = int(max(lens[lens_index], 0))
        s, e = map(int, seg)

        if s > N:
            s = N - 1; e = N
        else:
            if e > N: e = N
            if e <= s:
                s = min(max(s, 0), N - 1)
                e = s + 1

        vals = sig[s:e].tolist()
        parts.append(",".join(map(str, vals)))
        dwell_padded.append(v_dwell)
        lens_index += 1
        
    parts, dwell_padded = reverse_non_gap_segments(
        parts,
        dwell_padded,
        ref_aligned_sequence,
        na_token=na_token,
    )

    if empty_idx:
        print(
            f"Warning: empty signal segments at positions {empty_idx} "
            f"in read with ref_aligned_sequence length {len(ref_aligned_sequence)}"
        )

    signal_str = "|".join(parts)
    dwell_str  = ",".join(map(str, dwell_padded))
    return signal_str, dwell_str


def main(args):
    
    global REF_DICT
    REF_DICT = load_fasta_dict(args.ref)
    
    with pysam.AlignmentFile(args.bam, "rb") as bam:
        zstd_stream, raw_out = write_zstd(args.out)
        out = zstd_stream
                
        try:
            out.write("read_id\tchrom\tstart\tref_sequence\tqualities\tsequence\tdwell_time\tsignal\n")
            if args.workers and args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                    initargs=(args.pod5, REF_DICT),) as ex:
                    tasks = iter_tasks(bam, REF_DICT)
                    for line in tqdm(
                        ex.map(process_read, tasks, chunksize=256),
                        total=None,
                        unit="reads",
                        desc="Processing BAM",
                        dynamic_ncols=False,
                        mininterval=60,
                    ):
                        if line:
                            out.write(f"{line}\n")
            else:
                with pod5.DatasetReader(args.pod5) as p5:
                    it = bam.fetch(until_eof=True)
                    for rec in tqdm(
                        it,
                        total=None,
                        unit="reads",
                        desc="Processing BAM",
                        dynamic_ncols=False,
                        mininterval=60,
                    ):
                        if rec.is_unmapped or rec.is_secondary or rec.is_supplementary:
                            continue
                        chrom = rec.reference_name
                        if chrom not in REF_DICT:
                            continue
                        if not rec.has_tag("ur") or not rec.has_tag("ul"):
                            continue

                        read_id = rec.query_name.split()[0]

                        try:
                            ur_tag = rec.get_tag("ur")
                            ul_tag = rec.get_tag("ul")
                            
                            ur_start = get_ur_start(ur_tag)
                            if ur_start is None:
                                continue
                            start_pos_1based = ur_start + 1

                            pod5_read = p5.get_read(read_id)
                            sig = getattr(pod5_read, "signal", None)
                            if sig is None:
                                sig = getattr(pod5_read, "raw_signal", None)
                            if sig is None:
                                continue
                            
                            sig_arr = np.asarray(sig, dtype=np.int32)
                            ref_aligned, read_aligned, qual_str = \
                                extract_aligned_sequences_for_ur_region(
                                    chrom,
                                    rec.reference_start,
                                    rec.cigartuples,
                                    rec.query_sequence,
                                    rec.query_qualities,
                                    REF_DICT,   
                                    ur_tag,
                                )
                            if ref_aligned is None:
                                continue

                            signal, dwell_str = extract_signal(ul_tag, sig_arr, ref_aligned)

                            line = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (
                                read_id,
                                chrom,
                                start_pos_1based,
                                ref_aligned,
                                qual_str,
                                read_aligned,
                                dwell_str,
                                signal,
                            )
                            out.write(f"{line}\n")

                        except Exception as e:
                            print(f"Error processing read {read_id}: {e}", file=sys.stderr)
                            continue
        finally:
            out.close()
            if raw_out is not None:
                stream, fh = raw_out
                stream.close()
                fh.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAM + POD5 + FASTA -> TSV (manual arguments required)")
    parser.add_argument("--bam",  required=True,  help="Input BAM file path")
    parser.add_argument("--pod5", required=True,  help="Input POD5 path (file or directory)")
    parser.add_argument("--ref",  required=True,  help="Reference FASTA file path")
    parser.add_argument("--out",  required=True,  help="Output zstd-compressed TSV file path (.tsv.zst)")
    parser.add_argument("--workers",type=int,default=4,help="",)

    args = parser.parse_args()

    bam_p,pod5_p,ref_p,out_p  = Path(args.bam), Path(args.pod5), Path(args.ref), Path(args.out)

    for label, p in [("BAM", bam_p), ("POD5", pod5_p), ("FASTA", ref_p)]:
        if not p.exists():
            parser.error(f"{label} file not found: {p}")
    if out_p.parent and not out_p.parent.exists():
        parser.error(f"Output directory does not exist: {out_p.parent}")
    main(args)
