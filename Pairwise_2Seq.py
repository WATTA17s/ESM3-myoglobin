from Bio import pairwise2
from Bio.pairwise2 import format_alignment

def compare_with_alignment(ref_seq, mut_seq):
    alignments = pairwise2.align.globalxx(ref_seq, mut_seq)
    best = alignments[0]

    aligned_ref, aligned_mut, score, start, end = best

    mutations = []
    pos = 0  # position in reference (no gaps)

    for r, m in zip(aligned_ref, aligned_mut):
        if r != "-":
            pos += 1

        if r != m:
            if r == "-":
                mutations.append(f"ins{pos}{m}")
            elif m == "-":
                mutations.append(f"del{pos}{r}")
            else:
                mutations.append(f"{r}{pos}{m}")

    return aligned_ref, aligned_mut, mutations


# Example
ref = "M"
mut = "M"

aligned_ref, aligned_mut, mutations = compare_with_alignment(ref, mut)

print(aligned_ref)
print(aligned_mut)
print(mutations)
