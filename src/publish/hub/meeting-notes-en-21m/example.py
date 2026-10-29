"""Meeting notes with meeting-notes-en-21m.

pip install "keras>=3.6" jax tokenizers huggingface_hub
python example.py                       # downloads the model from the Hub
python example.py path/to/local/folder  # or uses a local copy

The second example also downloads <hf-user>/punctuation-casing-en-7m, which restores
punctuation and casing of a raw (lowercase, unpunctuated) transcript first.
"""

import importlib.util
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "<hf-user>/meeting-notes-en-21m"
CLEANUP_REPO_ID = "<hf-user>/punctuation-casing-en-7m"


def load_module(folder: Path, name: str):
    """Import ``folder/modeling.py`` under ``name`` (both repos ship a ``modeling.py``)."""
    spec = importlib.util.spec_from_file_location(name, folder / "modeling.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(snapshot_download(REPO_ID))
notes = load_module(folder, "notes_modeling")
notetaker = notes.NoteTaker.from_folder(folder)

meeting = """\
Anna: Thanks for joining. We need to decide on the launch date today.
Ben: The app is almost ready, but the payment page still has two bugs.
Anna: How long do you need to fix them?
Ben: About a week, if nobody pulls me onto other work.
Chloe: Marketing can be ready by the 15th, but not earlier.
Anna: Then let's launch on the 15th. Ben, please fix the payment bugs by Friday.
Ben: Okay, I'll send an update on Wednesday.
Chloe: I'll book the newsletter for the morning of the 15th.
Anna: Great. Anything else? No? Thanks, everyone."""
print(notetaker(meeting))

# A raw speech-to-text transcript: restore punctuation and casing first.
raw = """\
a: so did you talk to the landlord about the heating
b: yes he said someone will come on monday to fix it
a: monday is fine but we need to be home right
b: one of us has to be there between nine and twelve
a: i can work from home that day
b: great then i will tell him to come at ten
a: and ask him about the rent increase too
b: ok i will email him tonight"""
cleanup_folder = Path(snapshot_download(CLEANUP_REPO_ID))
cleaner = load_module(cleanup_folder, "cleanup_modeling").Cleaner.from_folder(cleanup_folder)
print(notetaker(raw, clean=cleaner))
