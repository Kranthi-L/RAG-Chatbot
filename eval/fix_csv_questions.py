# eval/fix_csv_questions.py
import io, sys, os, csv

IN_PATH  = sys.argv[1] if len(sys.argv) > 1 else "eval/networking_eval.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "eval/networking_eval_corrected.csv"

def main():
    # read all lines as raw text (UTF-8)
    with open(IN_PATH, "r", encoding="utf-8") as f:
        raw = f.read().splitlines()

    cleaned = []
    header_needed = True

    # If the first non-comment/non-blank line already looks like a header, don't add another
    for line in raw:
        if not line.strip() or line.lstrip().startswith("#"):
            cleaned.append(line)
            continue
        # first data-ish line encountered
        first = line.strip()
        looks_like_header = all(k in first.split(",") for k in ["question","gpt_response","claude_response","ideal_answer"])
        header_needed = not looks_like_header
        break

    # Re-scan with a flag to know if we've already passed the initial non-comment line
    cleaned = []
    have_written_header = False
    for line in raw:
        if not line.strip():
            cleaned.append(line)
            continue
        if line.lstrip().startswith("#"):
            cleaned.append(line)
            continue

        if not have_written_header:
            if header_needed:
                cleaned.append("question,gpt_response,claude_response,ideal_answer")
            have_written_header = True

        # Now we're on a data line. We expect format like:
        # <question>,,<gpt>,<claude>,"<ideal_answer>"
        # In your current file gpt/claude are empty, so it's usually:
        # <question>,,,"<ideal_answer>"
        # If <question> contains commas, it must be quoted.
        # Strategy: split on the first occurrence of ",,," (three commas), keeping the right side intact.
        idx = line.find(",,,")
        if idx == -1:
            # If there isn't a sentinel ",,,", try to split into 4 parts once, preserving the last field
            parts = line.split(",", 3)
            if len(parts) == 4:
                q, g, c, ideal = parts
            else:
                # Fallback: treat everything before the last quoted block as question
                # Try to find the start of the quoted ideal answer
                qmark = line.find(',,"')
                if qmark == -1:
                    qmark = line.find(',"')
                if qmark != -1:
                    q = line[:qmark]
                    rest = line[qmark+1:]  # keep leading comma for csv join below
                    # Force quoting for q
                    q = '"' + q.replace('"', '""') + '"'
                    cleaned.append(q + rest)
                    continue
                else:
                    # As a last resort, quote entire line as question and add empty cols
                    q = '"' + line.replace('"', '""') + '"'
                    cleaned.append(f'{q},,,')
                    continue
        else:
            q = line[:idx]
            rest = line[idx+3:]  # starts after the three commas

            # Force-quote the question (escape internal quotes)
            q = '"' + q.replace('"', '""') + '"'
            cleaned.append(q + ",,," + rest)
            continue

        # Normal path when we had split(",", 3)
        q = '"' + q.replace('"', '""') + '"'  # quote question
        # Ensure g/c are present; if they have commas, they should be quoted too
        g = '"' + g.replace('"', '""') + '"' if g and ("," in g or '"' in g) else g
        c = '"' + c.replace('"', '""') + '"' if c and ("," in c or '"' in c) else c
        # Ideal might already be quoted; if not, quote it
        ideal_stripped = ideal.strip()
        if not (ideal_stripped.startswith('"') and ideal_stripped.endswith('"')):
            ideal = '"' + ideal.replace('"', '""') + '"'
        cleaned.append(",".join([q, g, c, ideal]))

    # Write out the corrected CSV
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        for i, line in enumerate(cleaned):
            f.write(line + ("\n" if i < len(cleaned)-1 or cleaned[-1] != "" else ""))

    # Validate by parsing with Python's csv module
    data_rows = []
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row or (row and row[0].startswith("#")):
                continue
            if header is None:
                header = row
                continue
            if len(row) != 4:
                raise ValueError(f"Row has {len(row)} columns, expected 4: {row[:2]}...")
            data_rows.append(row)

    print(f"Fixed CSV saved → {OUT_PATH}  (rows: {len(data_rows)})")

if __name__ == "__main__":
    main()
