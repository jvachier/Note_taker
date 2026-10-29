"""Restore punctuation and casing with punctuation-casing-en-7m.

pip install "keras>=3.6" jax tokenizers huggingface_hub
python example.py                       # downloads the model from the Hub
python example.py path/to/local/folder  # or uses a local copy
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "<hf-user>/punctuation-casing-en-7m"

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(snapshot_download(REPO_ID))
sys.path.insert(0, str(folder))  # modeling.py ships with the model

import modeling  # noqa: E402

cleaner = modeling.Cleaner.from_folder(folder)
text = (
    "okay so lets start with the budget we spent about twenty percent more than planned on "
    "marketing in march john do you know why that happened i think it was the new campaign on "
    "instagram we should check the numbers with the finance team next week"
)
print(cleaner(text))
