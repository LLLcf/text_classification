```
conda create -n nlp_env python=3.10 -y
conda activate nlp_env
pip install ipykernel
python -m ipykernel install --name nlp_env
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0
pip install pandas numpy torch torchvision torchaudio transformers scikit-learn tqdm
```