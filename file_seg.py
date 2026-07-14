import json
import os
import glob
import re

# Thư mục chứa các file json
input_dir = "."
output_dir = "."

# Lấy tất cả file json
json_files = glob.glob(os.path.join(input_dir, "ocr_gemma-4-31b_nlvnpf-*_results.json"))

for json_file in sorted(json_files):

    filename = os.path.basename(json_file)

    # Lấy mã tác phẩm
    # Ví dụ:
    # ocr_gemma-4-31b_nlvnpf-0250-001_results.json -> nlvnpf-0250
    # ocr_gemma-4-31b_nlvnpf-0475-01-001_results.json -> nlvnpf-0475-01
    m = re.match(r"ocr_gemma-4-31b_(nlvnpf-.+)-001_results\.json$", filename)

    if m is None:
        print(f"Bỏ qua: {filename}")
        continue

    prefix = m.group(1)

    # Tên file đầu ra
    output_file = os.path.join(output_dir, f"{prefix}_seg.tsv")

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(data, start=1):
            text = item.get("result", "").replace("\n", " ").strip()
            f.write(f"{prefix}_{i:06d}\t{text}\n")

    print(f"✓ {filename} -> {os.path.basename(output_file)}")

print("Hoàn thành!")