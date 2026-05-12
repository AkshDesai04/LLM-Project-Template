import os
import sys

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.markitdown_utils import MarkItDownUtils
from utils.logger import get_logger

logger = get_logger("TestMarkItDown")

def test_conversions():
    md_utils = MarkItDownUtils()
    
    test_files = [
        "tests/test_files/doggo.jpg",
        "tests/test_files/Lorem_Ipsum.pdf"
    ]
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            logger.warning(f"Skipping {file_path}, file not found.")
            continue
            
        try:
            logger.info(f"Testing conversion of: {file_path}")
            markdown_content = md_utils.convert_local(file_path)
            
            print(f"\n--- Markdown Content for {file_path} ---")
            # Print first 500 characters
            print(markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content)
            print("-" * 40)
            
        except Exception as e:
            logger.error(f"Test failed for {file_path}: {e}")

if __name__ == "__main__":
    test_conversions()
