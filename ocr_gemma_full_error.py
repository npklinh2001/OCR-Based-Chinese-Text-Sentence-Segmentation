from __future__ import annotations

import base64
import json
import mimetypes
import urllib.request
from pathlib import Path

from cerebras.cloud.sdk import Cerebras

MODEL = "gemma-4-31b"

API_KEYS = [
    "csk-y233medkv2m29x3dfhdxd6jpnhwcnfxjeh4td3pkymhjecww", #jamejordan
    "csk-c8e235mwe8mp5tx4nrn53xhymwx3v2kt4crckfd986eh9r9p", #lekhanhphuong
    "csk-trkjhp6hxc3yrxcmmvcdpde4wm5x5penpf8y5e6r3jkdwnfh",
    "csk-6cv3hk46vv4wj8eh2j5j4df949mkt82n65mfkp9pk59532hp", # nguyenphankhanhlinh2001
    "csk-6jrvcmmdp58vmrxfmc4tn3kr5ph3wn66pt2r35dw23xr9pkr", # npkl01112001
    "csk-cx8xrw24m3998y636rkjr9fxx5vmen8452dfecjee82d8crc" # ngphkhanhlinh2001
]


OCR_PROMPT = (
    "OCR this image. Return only the text visible in the image. "
    "Preserve line breaks. Do not translate, explain, or add notes. "
    "For traditional Chinese or Han documents written vertically, "
    "Read from the rightmost column first, read each column from top to bottom, "
    "then continue to the next column on the left. "
    "If the text is Han or Chinese characters, keep the original characters."
)


def download_image(image_url: str):
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


def call_cerebras(image_url: str, api_key: str):
    image_bytes, mime_type = download_image(image_url)

    client = Cerebras(api_key=api_key)

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
                            "url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


def retry_json(json_path: Path):
    print(f"\n===== {json_path.name} =====")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    key_index = 0
    retry_count = 0
    success_count = 0

    for item in data:

        # Chỉ retry những dòng có error
        if "error" not in item:
            continue

        retry_count += 1

        print(f"Retry: {item['url']}")

        try:
            result = call_cerebras(
                item["url"],
                API_KEYS[key_index],
            )

            item.pop("error", None)
            item["result"] = result

            success_count += 1

        except Exception as e:
            item["error"] = str(e)
            print(e)

        key_index = (key_index + 1) % len(API_KEYS)

        # lưu ngay sau mỗi lần OCR
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Retry: {retry_count}")
    print(f"Success: {success_count}")


def main():

    json_files = sorted(Path(".").glob("ocr_gemma-4-31b_*_results.json"))

    print(f"Tìm thấy {len(json_files)} file.")

    for json_file in json_files:
        retry_json(json_file)


if __name__ == "__main__":
    main()