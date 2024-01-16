conda install -y rdkit=2020.03.3 -c rdkit 
conda install pytorch==1.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install joblib networkx graphviz pyaml pynvml tqdm torchtext==0.6.0 configargparse
pip install -e onmt/