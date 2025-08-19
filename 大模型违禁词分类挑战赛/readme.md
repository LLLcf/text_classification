```
conda create -n nlp_env python=3.10 -y
conda activate nlp_env
pip install ipykernel
python -m ipykernel install --name nlp_env
# pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install pandas numpy torch torchvision torchaudio transformers==4.43 scikit-learn tqdm
```

```
conda create -n llm_env python=3.10 -y
conda activate llm_env
pip install ipykernel
python -m ipykernel install --name llm_env
# pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
pip install transformers==4.51.1 sentence-transformers==2.7.0 vllm==0.8.5 scikit-learn tqdm pandas numpy
```

