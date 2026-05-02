# 1. Welche CUDA-Libs sieht das pixi-Environment?
pixi run python -c "
import ctypes, os
try:
    cuda = ctypes.CDLL('libcuda.so.1')
    print('libcuda.so.1 geladen')
except: print('libcuda.so.1 NICHT gefunden')

# CatBoost's interne CUDA-Version
from catboost.utils import get_gpu_device_count
print('GPU count:', get_gpu_device_count())
"

# 2. Gibt es konfliktende CUDA-Pakete in pixi?
pixi list | grep -i cuda

# 3. LD_LIBRARY_PATH prüfen
pixi run env | grep -i LD_LIBRARY_PATH

# 4. Welche libcuda wird tatsächlich gelinkt?
pixi run python -c "
import catboost._catboost as cb
import subprocess
pid = str(subprocess.os.getpid())
result = subprocess.run(['cat', f'/proc/{pid}/maps'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'cuda' in line.lower() or 'nvidia' in line.lower():
        print(line.split()[-1] if len(line.split()) > 5 else line)
"
