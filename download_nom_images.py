from __future__ import annotations

from pathlib import Path
from time import sleep
import urllib.error
import urllib.request

import pandas as pd

EXCEL_PATH = Path("DS Đề tài KHDL.xlsx")
SHEET_NAME = "Tách trang"
DOWNLOAD_ROOT = Path("downloaded_images")
IMAGE_URL_TEMPLATE = (
    "https://lib.nomfoundation.org/site_media/nom/{vnpf_id}/jpeg/{vnpf_id}-{page:03d}.jpg"
)


def read_page_number(value: object, column_name: str, row_number: int) -> int:
    if pd.isna(value):
        raise ValueError(f"Dong {row_number} thieu gia tri cot {column_name}.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Dong {row_number} cot {column_name} khong phai so trang hop le: {value!r}"
        ) from exc

    if not numeric_value.is_integer():
        raise ValueError(
            f"Dong {row_number} cot {column_name} phai la so nguyen: {value!r}"
        )

    page = int(numeric_value)
    if page < 1:
        raise ValueError(f"Dong {row_number} cot {column_name} phai >= 1.")

    return page


def load_download_jobs(
    excel_path: str | Path = EXCEL_PATH,
    sheet_name: str = SHEET_NAME,
) -> list[dict[str, object]]:
    data = pd.read_excel(excel_path, sheet_name=sheet_name)
    data.columns = data.columns.astype(str).str.strip()

    required_columns = ["VNPF Id", "FROM", "TO"]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Thieu cot trong file Excel: {', '.join(missing_columns)}")

    jobs: list[dict[str, object]] = []
    for index, row in data.iterrows():
        row_number = index + 2
        if pd.isna(row["VNPF Id"]):
            continue

        vnpf_id = str(row["VNPF Id"]).strip()
        if not vnpf_id:
            continue

        jobs.append(
            {
                "vnpf_id": vnpf_id,
                "from_page": read_page_number(row["FROM"], "FROM", row_number),
                "to_page": read_page_number(row["TO"], "TO", row_number),
            }
        )

    return jobs


def build_image_url(vnpf_id: str, page: int) -> str:
    return IMAGE_URL_TEMPLATE.format(vnpf_id=vnpf_id, page=page)


def build_image_path(vnpf_id: str, page: int) -> Path:
    return DOWNLOAD_ROOT / vnpf_id / f"{vnpf_id}-{page:03d}.jpg"


def download_image(url: str, output_path: Path, retries: int = 3) -> bool:
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Bo qua da co: {output_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "download-nom-images/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                image_bytes = response.read()

            output_path.write_bytes(image_bytes)
            print(f"Da tai: {output_path}")
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                print(f"Loi tai {url}: {exc}")
                return False

            print(f"Thu lai {attempt}/{retries}: {url} ({exc})")
            sleep(2)

    return False


def main() -> None:
    jobs = load_download_jobs()
    total_files = sum(
        int(job["to_page"]) - int(job["from_page"]) + 1
        for job in jobs
    )
    downloaded_count = 0

    print(f"Doc duoc {len(jobs)} dong tu {EXCEL_PATH}")
    print(f"Can tai toi da {total_files} anh vao thu muc {DOWNLOAD_ROOT}")

    for job in jobs:
        vnpf_id = str(job["vnpf_id"])
        from_page = int(job["from_page"])
        to_page = int(job["to_page"])

        if from_page > to_page:
            raise ValueError(f"{vnpf_id}: FROM phai <= TO.")

        print(f"Dang tai {vnpf_id}: trang {from_page}-{to_page}")
        for page in range(from_page, to_page + 1):
            url = build_image_url(vnpf_id, page)
            output_path = build_image_path(vnpf_id, page)
            if download_image(url, output_path):
                downloaded_count += 1

    print(f"Xong. Tai moi {downloaded_count}/{total_files} anh.")


if __name__ == "__main__":
    main()
