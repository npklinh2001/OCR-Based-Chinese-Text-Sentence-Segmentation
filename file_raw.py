import json
import os
import glob
import re

# ==========================
# Cấu hình
# ==========================
input_dir = "."      # Thư mục chứa file JSON
output_dir = "."     # Thư mục lưu file RAW

# Lấy tất cả file JSON
json_files = glob.glob(
    os.path.join(input_dir, "ocr_gemma-4-31b_nlvnpf-*_results.json")
)

if not json_files:
    print("Không tìm thấy file JSON nào!")
    exit()

for json_file in sorted(json_files):

    filename = os.path.basename(json_file)

    # Ví dụ:
    # ocr_gemma-4-31b_nlvnpf-0250-001_results.json
    # -> nlvnpf-0250
    #
    # ocr_gemma-4-31b_nlvnpf-0475-01-001_results.json
    # -> nlvnpf-0475-01
    m = re.match(
        r"ocr_gemma-4-31b_(nlvnpf-.+)-001_results\.json$",
        filename
    )

    if not m:
        print(f"Bỏ qua: {filename}")
        continue

    work_id = m.group(1)

    output_path = os.path.join(
        output_dir,
        f"{work_id}_raw.txt"
    )

    # Đọc JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Nếu chỉ có 1 object thì chuyển thành list
    if isinstance(data, dict):
        data = [data]

    with open(output_path, "w", encoding="utf-8") as out:

        for item in data:

            # Lấy tên ảnh
            image_name = os.path.splitext(
                os.path.basename(item["url"])
            )[0]

            # Giữ nguyên toàn bộ xuống dòng của OCR
            text = item.get("result", "")

            lines = []
            for line in text.splitlines():
                # đổi khoảng trắng toàn chiều rộng thành space
                line = line.replace("\u3000", " ")

                # nếu cả dòng chỉ là khoảng trắng thì bỏ
                if line.strip() == "":
                    continue

                lines.append(line.rstrip())

            text = "\n".join(lines)

            # Ghi block
            out.write(f"### {image_name} ###\n\n")
            out.write(text)
            out.write("\n\n")

    print(f"✓ {filename} -> {os.path.basename(output_path)}")

print("\nHoàn thành!")