from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from cerebras.cloud.sdk import Cerebras

MODEL = "gemma-4-31b"  # đổi sang model vision khác trên Cerebras nếu cần

FIRST_IMAGE_URL = "https://lib.nomfoundation.org/site_media/nom/nlvnpf-0503/jpeg/nlvnpf-0503-001.jpg"
FROM_PAGE = 21
TO_PAGE = 60
OUTPUT_PATH = r"C:\\Users\\ASUS\\Downloads\\midterm\\ocr_cerebras_results.json"

OCR_PROMPT = (
    "OCR this image. Return only the text visible in the image. "
    "Preserve line breaks. Do not translate, explain, or add notes. "
    "If the text is Han Nom or Chinese characters, keep the original characters."
)

client = Cerebras(api_key="csk-c8e235mwe8mp5tx4nrn53xhymwx3v2kt4crckfd986eh9r9p")


def download_image(image_url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(
        image_url,
        headers={"User-Agent": "ocr-nom/1.0"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        image_bytes = response.read()
        content_type = response.headers.get("Content-Type", "")

    mime_type = content_type.split(";", 1)[0].strip()
    if not mime_type:
        mime_type = mimetypes.guess_type(image_url)[0] or "image/jpeg"

    return image_bytes, mime_type


def call_cerebras_for_url(image_url: str) -> str:
    image_bytes, mime_type = download_image(image_url)
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


def ocr_image_urls(
    image_urls: list[str],
    output_path: str | Path,
    ocr_func=call_cerebras_for_url,
) -> list[dict[str, str]]:
    output_file = Path(output_path)
    saved_results = load_json_results(output_file)
    results = []

    for image_url in image_urls:
        try:
            result = ocr_func(image_url)
            entry = {"url": image_url, "result": result}
        except Exception as exc:
            entry = {"url": image_url, "error": str(exc)}

        results.append(entry)
        saved_results.append(entry)
        write_json_results(output_file, saved_results)

    return results


def load_json_results(output_file: Path) -> list[dict[str, str]]:
    if not output_file.exists():
        return []

    text = output_file.read_text(encoding="utf-8").strip()
    if not text:
        return []

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"File JSON output phải là một list: {output_file}")

    return data


def write_json_results(output_file: Path, results: list[dict[str, str]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_image_url_range(first_image_url: str, from_page: int, to_page: int) -> list[str]:
    if from_page > to_page:
        raise ValueError("from_page phải nhỏ hơn hoặc bằng to_page.")

    page_pattern = re.compile(r"-\d{3}(\.[A-Za-z0-9]+)$")
    if not page_pattern.search(first_image_url):
        raise ValueError("URL cần kết thúc bằng dạng -001.jpg, -002.jpg, ...")

    return [
        page_pattern.sub(lambda match: f"-{page:03d}{match.group(1)}", first_image_url)
        for page in range(from_page, to_page + 1)
    ]


def build_configured_image_urls() -> list[str]:
    return build_image_url_range(FIRST_IMAGE_URL, FROM_PAGE, TO_PAGE)


def main() -> None:
    image_urls = build_configured_image_urls()

    for image_url in image_urls:
        print(f"Đang OCR: {image_url}")

    results = ocr_image_urls(image_urls, OUTPUT_PATH)
    print(f"Đã lưu {len(results)} kết quả vào {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
