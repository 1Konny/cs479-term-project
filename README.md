# cs479-term-project


```
# Download data
bash download_data.sh

# Simple NFD with PointConv discriminator
python main.py --gen_type simple --disc_type pointconv --name simple_pointconv

# Simple NFD with Set Transformer discriminator
python main.py --gen_type simple --disc_type settsfm --name simple_settsfm 

# Modulation-based NFD with PointConv discriminator
python main.py --gen_type mod --disc_type pointconv --name mod_pointconv

# Modulation-based NFD with Set Transformer discriminator
python main.py --gen_type mod --disc_type settsfm --name mod_settsfm 
```
