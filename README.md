From windows powershell

# install python / git
```
winget install -e --id Python.Python.3.12
winget install Git.Git
```

# setup venv
close the window and open a new powershell
```
cd c:\users\$env:USERNAME
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\activate
```

# install torch
# for 50xx
```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
# for 40xx
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

# install mlmark
git clone https://github.com/pithetaxyz/mlmark.git
cd mlmark
pip install -r requirements.txt
```

# run benchmark
```
cd benchmarks
python run_all.py -o .
```
