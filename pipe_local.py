import os
from pathlib import Path
from collections import Counter

import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig

# =====================================
# CONFIG
# =====================================
MODEL_NAME = "esm3_sm_open_v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MASK_CHAR = "_"
WINDOW = 10
STEP = 10

N_SAMPLES = 20
TEMPERATURE = 0.5

FINALIST_THRESHOLD = 0.6

OUTPUT_DIR = Path("pipeline_text_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MASTER_LOG = OUTPUT_DIR / "full_pipeline_log.txt"


# =====================================
# INPUT
# =====================================
ref_seq = "PUT_YOUR_REFERENCE_SEQUENCE_HERE"

# ถ้ามี masked seq อยู่แล้ว ให้ใส่ list ตรงนี้ แล้วตั้ง USE_PREBUILT_MASKED_SEQS = True
masked_seqs = [
    # "__________AAAAAAAAAA....",
    # "AAAAAAAAAA__________....",
]

USE_PREBUILT_MASKED_SEQS = False


# =====================================
# FILE HELPERS
# =====================================
def save_text(path: Path, text: str):
    with open(path, "w") as f:
        f.write(text)


def save_lines(path: Path, lines: list[str]):
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def append_log(text: str):
    with open(MASTER_LOG, "a") as f:
        f.write(text)


# =====================================
# SEQ HELPERS
# =====================================
def build_masked_sequences(ref_seq: str, window: int = 10, step: int = 10, mask_char: str = "_") -> list[str]:
    seqs = []
    n = len(ref_seq)

    for start in range(0, n, step):
        end = min(start + window, n)
        seq = list(ref_seq)

        for i in range(start, end):
            seq[i] = mask_char

        seqs.append("".join(seq))

    return seqs


def masked_positions(seq: str, mask_char: str = "_") -> list[int]:
    return [i for i, ch in enumerate(seq) if ch == mask_char]


def generate_predictions(model, masked_seq: str, n_samples: int = 20, temperature: float = 0.7):
    preds = []

    with torch.no_grad():
        for i in range(n_samples):
            protein = ESMProtein(sequence=masked_seq)
            out = model.generate(
                protein,
                GenerationConfig(
                    track="sequence",
                    temperature=temperature
                )
            )
            preds.append(out.sequence)

    return preds


def consensus_line(masked_seq: str, preds: list[str], ref_seq: str, mask_char: str = "_") -> str:
    result = list(masked_seq)
    positions = masked_positions(masked_seq, mask_char)

    for pos in positions:
        aa_list = [p[pos] for p in preds]
        count = Counter(aa_list)
        top_aa, top_count = count.most_common(1)[0]

        if top_count == len(preds):  # 100% consensus
            if top_aa != ref_seq[pos]:
                result[pos] = mask_char
            else:
                result[pos] = ref_seq[pos]
        else:
            result[pos] = ref_seq[pos]

    return "".join(result)


def merge_consensus_lines(lines: list[str], ref_seq: str, mask_char: str = "_") -> str:
    n = len(ref_seq)
    merged = []

    for i in range(n):
        if any(line[i] == mask_char for line in lines):
            merged.append(mask_char)
        else:
            merged.append(ref_seq[i])

    return "".join(merged)


def build_three_stage_inputs(final_merged: str, ref_seq: str):
    n = len(final_merged)
    mid = n // 2

    full_seq = final_merged
    front_half_seq = final_merged[:mid] + ref_seq[mid:]
    back_half_seq = ref_seq[:mid] + final_merged[mid:]

    return {
        "full": full_seq,
        "front_half": front_half_seq,
        "back_half": back_half_seq,
    }


def build_finalist_seq(
    final_merged: str,
    ref_seq: str,
    all_stage_preds: list[str],
    threshold: float = 0.6,
    mask_char: str = "_",
):
    result = list(ref_seq)

    for pos, ch in enumerate(final_merged):
        if ch == mask_char:
            aa_list = [seq[pos] for seq in all_stage_preds]
            count = Counter(aa_list)

            top_aa, top_count = count.most_common(1)[0]
            frac = top_count / len(all_stage_preds)

            if frac >= threshold and top_aa != ref_seq[pos]:
                result[pos] = top_aa
            else:
                result[pos] = ref_seq[pos]
        else:
            result[pos] = ref_seq[pos]

    return "".join(result)


# =====================================
# MAIN
# =====================================
def main():
    # reset master log
    save_text(MASTER_LOG, "")

    append_log("===== PIPELINE START =====\n\n")
    append_log("REFERENCE SEQ:\n")
    append_log(ref_seq + "\n\n")

    # 1) prepare masked seqs
    if USE_PREBUILT_MASKED_SEQS:
        current_masked_seqs = masked_seqs
    else:
        current_masked_seqs = build_masked_sequences(
            ref_seq=ref_seq,
            window=WINDOW,
            step=STEP,
            mask_char=MASK_CHAR
        )

    save_lines(OUTPUT_DIR / "step1_masked_seqs.txt", current_masked_seqs)

    with open(OUTPUT_DIR / "step1_masked_seqs_numbered.txt", "w") as f:
        for i, seq in enumerate(current_masked_seqs, 1):
            f.write(f"[MASK {i:03d}]\n{seq}\n\n")

    append_log("STEP 1: MASKED SEQS\n")
    for i, seq in enumerate(current_masked_seqs, 1):
        append_log(f"[MASK {i:03d}] {seq}\n")
    append_log("\n")

    # 2) load model
    print("Loading model...")
    model = ESM3.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    print("Model loaded!")

    append_log("MODEL LOADED\n")
    append_log(f"MODEL_NAME: {MODEL_NAME}\n")
    append_log(f"DEVICE: {DEVICE}\n\n")

    # 3) per-mask predictions + consensus
    all_lines = []
    all_preds = []

    for i, masked_seq in enumerate(current_masked_seqs, 1):
        print(f"[{i}/{len(current_masked_seqs)}] Running mask...")

        preds = generate_predictions(
            model=model,
            masked_seq=masked_seq,
            n_samples=N_SAMPLES,
            temperature=TEMPERATURE
        )

        line = consensus_line(
            masked_seq=masked_seq,
            preds=preds,
            ref_seq=ref_seq,
            mask_char=MASK_CHAR
        )

        all_preds.append(preds)
        all_lines.append(line)

        # separated file per mask
        with open(OUTPUT_DIR / f"step2_mask_{i:03d}_details.txt", "w") as f:
            f.write(f"[MASK {i:03d}]\n")
            f.write("INPUT MASKED SEQ:\n")
            f.write(masked_seq + "\n\n")

            f.write("PREDICTIONS:\n")
            for j, pred in enumerate(preds, 1):
                f.write(f"{j:02d}: {pred}\n")

            f.write("\nCONSENSUS LINE:\n")
            f.write(line + "\n")

        # append to master log
        append_log(f"STEP 2: MASK {i:03d}\n")
        append_log("INPUT MASKED SEQ:\n")
        append_log(masked_seq + "\n")
        append_log("PREDICTIONS:\n")
        for j, pred in enumerate(preds, 1):
            append_log(f"{j:02d}: {pred}\n")
        append_log("CONSENSUS LINE:\n")
        append_log(line + "\n\n")

    # 4) save all consensus lines + final merged
    with open(OUTPUT_DIR / "step3_all_consensus_lines.txt", "w") as f:
        for i, line in enumerate(all_lines, 1):
            f.write(f"[CONSENSUS {i:03d}]\n{line}\n\n")

    final_merged = merge_consensus_lines(all_lines, ref_seq, mask_char=MASK_CHAR)
    save_text(OUTPUT_DIR / "step4_final_merged.txt", final_merged)

    append_log("STEP 3: ALL CONSENSUS LINES\n")
    for i, line in enumerate(all_lines, 1):
        append_log(f"[CONSENSUS {i:03d}] {line}\n")
    append_log("\n")

    append_log("STEP 4: FINAL MERGED\n")
    append_log(final_merged + "\n\n")

    # 5) build 3 stage inputs
    stage_inputs = build_three_stage_inputs(final_merged, ref_seq)

    with open(OUTPUT_DIR / "step5_stage_inputs.txt", "w") as f:
        for stage_name, seq in stage_inputs.items():
            f.write(f"[{stage_name}]\n{seq}\n\n")

    append_log("STEP 5: STAGE INPUTS\n")
    for stage_name, seq in stage_inputs.items():
        append_log(f"[{stage_name}] {seq}\n")
    append_log("\n")

    # 6) run 3 stages, each 20 predictions
    all_stage_preds_dict = {}
    all_stage_preds_flat = []

    for stage_name, seq in stage_inputs.items():
        print(f"Running stage: {stage_name}")

        preds = generate_predictions(
            model=model,
            masked_seq=seq,
            n_samples=N_SAMPLES,
            temperature=TEMPERATURE
        )

        all_stage_preds_dict[stage_name] = preds
        all_stage_preds_flat.extend(preds)

        with open(OUTPUT_DIR / f"step6_stage_{stage_name}_predictions.txt", "w") as f:
            f.write(f"[STAGE: {stage_name}]\n")
            f.write("INPUT:\n")
            f.write(seq + "\n\n")
            f.write("PREDICTIONS:\n")
            for i, pred in enumerate(preds, 1):
                f.write(f"{i:02d}: {pred}\n")

        append_log(f"STEP 6: STAGE {stage_name}\n")
        append_log("INPUT:\n")
        append_log(seq + "\n")
        append_log("PREDICTIONS:\n")
        for i, pred in enumerate(preds, 1):
            append_log(f"{i:02d}: {pred}\n")
        append_log("\n")

    # 7) build finalist seq
    finalist_seq = build_finalist_seq(
        final_merged=final_merged,
        ref_seq=ref_seq,
        all_stage_preds=all_stage_preds_flat,
        threshold=FINALIST_THRESHOLD,
        mask_char=MASK_CHAR
    )

    save_text(OUTPUT_DIR / "step7_finalist_seq.txt", finalist_seq)

    append_log("STEP 7: FINALIST SEQ\n")
    append_log(finalist_seq + "\n\n")

    # 8) summary
    with open(OUTPUT_DIR / "pipeline_summary.txt", "w") as f:
        f.write("REFERENCE SEQ:\n")
        f.write(ref_seq + "\n\n")

        f.write("FINAL MERGED:\n")
        f.write(final_merged + "\n\n")

        f.write("STAGE INPUTS:\n")
        for stage_name, seq in stage_inputs.items():
            f.write(f"{stage_name}: {seq}\n")
        f.write("\n")

        f.write("FINALIST SEQ:\n")
        f.write(finalist_seq + "\n")

    append_log("===== PIPELINE END =====\n")

    print("\nDONE")
    print("Outputs saved in:", OUTPUT_DIR.resolve())
    print("Final merged :", final_merged)
    print("Finalist seq :", finalist_seq)


if __name__ == "__main__":
    main()
