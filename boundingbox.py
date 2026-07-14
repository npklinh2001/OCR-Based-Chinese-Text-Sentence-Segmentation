"""
Chạy bounding-box detection cho TẤT CẢ ảnh trong thư mục downloaded_images
(bao gồm mọi thư mục con: nlvnpf-0250, nlvnpf-0251, nlvnpf-0475-01, ...)

- Duyệt đệ quy toàn bộ file ảnh (.jpg, .jpeg, .png, .tif, .tiff, .bmp)
- Ảnh nào xử lý xong thì lưu ngay lập tức (không đợi các ảnh khác)
- Tự động bỏ qua ảnh đã xử lý trước đó (chạy lại không bị lặp/mất thời gian)
- Ghi log tiến độ + log lỗi ra file riêng để dễ theo dõi khi chạy hàng ngàn ảnh
"""

import cv2
import easyocr
import numpy as np
from pathlib import Path
import time
import traceback

# ====== CẤU HÌNH ======
INPUT_ROOT = r"downloaded_images"          # thư mục gốc chứa các folder con nlvnpf-xxxx
OUTPUT_ROOT = r"downloaded_images_bbox"     # kết quả sẽ lưu theo đúng cấu trúc folder gốc
LANG_LIST = ['ch_tra']                      # đổi thành ['ch_tra','vi'] nếu ảnh có lẫn tiếng Việt
USE_GPU = True                             # đổi True nếu máy bạn có CUDA
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# =======================

LOG_FILE = Path(OUTPUT_ROOT) / "_progress_log.txt"
ERROR_LOG_FILE = Path(OUTPUT_ROOT) / "_error_log.txt"


def log(msg: str, path: Path = LOG_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def collect_images(root: str):
    """Duyệt đệ quy toàn bộ ảnh trong root, kể cả trong các folder con."""
    root_path = Path(root)
    return sorted(
        p for p in root_path.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )


def process_image(reader, image_path: Path, output_root: Path, input_root: Path):
    # Giữ nguyên cấu trúc thư mục con khi lưu kết quả
    relative = image_path.relative_to(input_root)
    out_path = output_root / relative.parent / f"{relative.stem}_bbox.jpg"
    out_json = output_root / relative.parent / f"{relative.stem}_bbox.json"

    # Bỏ qua nếu đã xử lý rồi (cho phép dừng giữa chừng rồi chạy lại)
    if out_path.exists():
        return "skipped"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = reader.readtext(str(image_path), paragraph=False)

    # Vẽ bbox lên ảnh
    boxes_data = []
    for (bbox, text, conf) in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(image_rgb, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        boxes_data.append({
            "bbox": [[float(x), float(y)] for x, y in bbox],
            "text": text,
            "confidence": float(conf),
        })

    # Lưu ảnh kết quả NGAY khi xong (không đợi ảnh khác)
    cv2.imwrite(str(out_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    # Lưu luôn dữ liệu bbox dạng JSON để dùng lại sau này (khỏi phải OCR lại)
    import json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(boxes_data, f, ensure_ascii=False, indent=2)

    return f"done ({len(results)} boxes)"


def main():
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    log("Đang khởi tạo EasyOCR reader...")
    reader = easyocr.Reader(LANG_LIST, gpu=USE_GPU)

    images = collect_images(input_root)
    log(f"Tìm thấy {len(images)} ảnh trong '{input_root}' (kể cả các thư mục con).")

    done_count = 0
    skip_count = 0
    error_count = 0

    for i, img_path in enumerate(images, start=1):
        try:
            status = process_image(reader, img_path, output_root, input_root)
            if status == "skipped":
                skip_count += 1
                log(f"[{i}/{len(images)}] BỎ QUA (đã có kết quả): {img_path}")
            else:
                done_count += 1
                log(f"[{i}/{len(images)}] XONG - {status}: {img_path}")
        except Exception as e:
            error_count += 1
            err_msg = f"[{i}/{len(images)}] LỖI: {img_path} -> {e}"
            log(err_msg, path=ERROR_LOG_FILE)
            log(traceback.format_exc(), path=ERROR_LOG_FILE)
            log(err_msg)  # cũng ghi vào log chính để thấy ngay trên console

    log(f"HOÀN TẤT. Thành công: {done_count} | Bỏ qua (đã xử lý trước): {skip_count} | Lỗi: {error_count}")
    log(f"Kết quả lưu tại: {output_root.resolve()}")


if __name__ == "__main__":
    main()