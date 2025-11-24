conda activate gptsovits
nohup python api_v2.py   -c GPT_SoVITS/configs/tts_infer.yaml   -a 0.0.0.0   -p 9880 > ../logs/api_v2.logs &