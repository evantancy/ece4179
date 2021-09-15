# ece4179_nndl
ECE4179 S2 2021

This repo contains all code for [ECE4179](https://handbook.monash.edu/2021/units/ECE4179?year=2021)

# Setup
First activate your python environment
```bash
cd ece4179_nndl/
pip install -r setup/requirements.txt
```

After that configure **nbstripout** to automatically remove output in jupyter notebooks when committing changes
```bash
nbstripout --install --attributes .gitattributes
```

See [link](http://humanscode.com/better-python-environment-management-for-anaconda) for automatic activation of conda environments