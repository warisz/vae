# vae
VAE for Stanford dogs dataset

### Instructions
Dependencies:
```pip install -r requirements.txt```

Download dataset from HF:
```python download_data.py```

Train model and output loss graph:
```python train.py```

Output reconstructions for 50 images:
```python reconstruct.py```


### Output

Reconstruction output for 50 random samples:
![Reconstructed Dogs](reconstruction.png)


![Loss Graph](loss_graph.png)
