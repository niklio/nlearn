import urllib.request
import os

os.makedirs("datasets", exist_ok=True)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
dest = "datasets/shakespeare.txt"

if os.path.exists(dest):
    print(f"{dest} already exists, skipping.")
else:
    print(f"Downloading {dest}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Done ({os.path.getsize(dest) // 1024}KB)")
