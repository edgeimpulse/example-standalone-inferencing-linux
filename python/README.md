How to run:

```bash
# Setup venv
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt

# Run inference
python3 run_inference.py --use-npu
# Runs fine w/o segfaulting
```
