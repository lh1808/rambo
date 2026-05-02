(generic) ubuntu@192.168.5.216 ~/rubin $ pixi run python -c "
import ctypes, os
try:
    cuda = ctypes.CDLL('libcuda.so.1')
    print('libcuda.so.1 geladen')
except: print('libcuda.so.1 NICHT gefunden')

# CatBoost's interne CUDA-Version
from catboost.utils import get_gpu_device_count
print('GPU count:', get_gpu_device_count())
"
libcuda.so.1 geladen
GPU count: 1
(generic) ubuntu@192.168.5.216 ~/rubin $ pixi list | grep -i cuda
catboost                            1.2.10        cuda129_py312h3fcdad1_100    44.61 MiB  conda  https://nexus3.lan.huk-coburg.de/repository/conda-forge
cuda-cudart                         12.9.79       h5888daf_0                   22.70 KiB  conda  https://nexus3.lan.huk-coburg.de/repository/conda-forge
cuda-cudart_linux-64                12.9.79       h3f2d84a_0                  192.63 KiB  conda  https://nexus3.lan.huk-coburg.de/repository/conda-forge
cuda-version                        12.9          h4f385c5_3                   21.07 KiB  conda  https://nexus3.lan.huk-coburg.de/repository/conda-forge
(generic) ubuntu@192.168.5.216 ~/rubin $ pixi run env | grep -i LD_LIBRARY_PATH
JAVA_LD_LIBRARY_PATH=/opt/conda/envs/generic/lib/jvm/lib/server
JAVA_LD_LIBRARY_PATH_BACKUP=/opt/conda/envs/generic/lib/jvm/lib/server
LD_LIBRARY_PATH=/opt/oracle/instantclient_21_6
(generic) ubuntu@192.168.5.216 ~/rubin $ pixi run python -c "
import catboost._catboost as cb
import subprocess
pid = str(subprocess.os.getpid())
result = subprocess.run(['cat', f'/proc/{pid}/maps'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'cuda' in line.lower() or 'nvidia' in line.lower():
        print(line.split()[-1] if len(line.split()) > 5 else line)
"
/mnt/rubin/.pixi/envs/default/targets/x86_64-linux/lib/libcudart.so.12.9.79
/mnt/rubin/.pixi/envs/default/targets/x86_64-linux/lib/libcudart.so.12.9.79
/mnt/rubin/.pixi/envs/default/targets/x86_64-linux/lib/libcudart.so.12.9.79
/mnt/rubin/.pixi/envs/default/targets/x86_64-linux/lib/libcudart.so.12.9.79
/mnt/rubin/.pixi/envs/default/targets/x86_64-linux/lib/libcudart.so.12.9.79
/mnt/rubin/.pixi/envs/default/lib/libicudata.so.78.3
/mnt/rubin/.pixi/envs/default/lib/libicudata.so.78.3
