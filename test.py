import subprocess
result = subprocess.run(["which", "tesseract"], capture_output=True, text=True)
print("Tesseract path:", result.stdout)