import pytesseract
from PIL import Image
import cv2
import numpy as np
import re

# Tell Python where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """Clean up the image so OCR reads it better"""
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to make text bigger and clearer
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Sharpen the image
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray

def extract_income_from_certificate(image_path):
    """
    Extract Annual Family Income from Kerala Government Income Certificate
    Example line: ₹.380000 (Rupees three lakh eighty thousand only)
    """
    print(f"Reading: {image_path}")

    # Preprocess image
    img = preprocess_image(image_path)

    # Run OCR
    text = pytesseract.image_to_string(img, lang="eng")

    print("\n--- OCR Raw Text ---")
    print(text)
    print("--------------------\n")

    # Find the income amount
    annual_income = find_income(text)
    return annual_income, text

def find_income(text):
    """
    Find income from Kerala certificate format
    Handles: ₹.380000 or Rs.380000 or ₹ 380000
    """

    # Pattern 1: ₹.380000 or ₹380000 (Kerala certificate style)
    patterns = [
        r'[₹Rs\.]+\s*\.?\s*([0-9,]+)',        # ₹.380000 or Rs.380000
        r'[Ii]ncome[^0-9]*([0-9,]+)',          # Income ... 380000
        r'[Aa]nnual[^0-9]*([0-9,]+)',          # Annual ... 380000
        r'is\s*[₹Rs\.]*\s*([0-9,]+)',          # "is ₹380000"
        r'([0-9]{5,7})',                        # Any 5-7 digit number
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            raw = match.replace(",", "").strip()
            try:
                amount = int(raw)
                # Must be realistic annual income (10000 to 50,00,000)
                if 10000 <= amount <= 5000000:
                    print(f"✅ Found annual income: ₹{amount}")
                    monthly = amount // 12
                    print(f"   Monthly income = ₹{amount} ÷ 12 = ₹{monthly}")
                    return amount
            except:
                continue

    print("⚠️ Could not extract income automatically")
    return None

# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter full path to certificate image: ").strip()

    annual_income, raw_text = extract_income_from_certificate(path)

    if annual_income:
        monthly_income = annual_income // 12
        print(f"\n{'='*40}")
        print(f"✅ Annual Income  : ₹{annual_income}")
        print(f"✅ Monthly Income : ₹{monthly_income}")
        print(f"{'='*40}")
    else:
        print("\n⚠️ Could not read income. You can enter it manually in the app.")