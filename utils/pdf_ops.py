from io import BytesIO
from typing import Optional, BinaryIO

from PyPDF2 import PdfReader, PdfWriter


def split_pages(input_stream: BinaryIO, start_index: Optional[int] = None, end_index: Optional[int] = None) -> BytesIO:
    pdf_reader = PdfReader(input_stream)
    pdf_writer = PdfWriter()
    total_pages = len(pdf_reader.pages)

    # Resolve default values and clamp to valid range
    start_idx = 0 if start_index is None else max(0, min(start_index, total_pages - 1))
    end_idx = total_pages - 1 if end_index is None else max(start_idx, min(end_index, total_pages - 1))

    for page_index in range(start_idx, end_idx + 1):
        pdf_writer.add_page(pdf_reader.pages[page_index])

    output_stream = BytesIO()
    pdf_writer.write(output_stream)
    output_stream.seek(0)

    return output_stream

def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    try:
        pdf_file_obj = BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file_obj)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF bytes: {e}")