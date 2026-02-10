#conda install -c conda-forge uvicorn==0.34.0 PyYAML==6.0.2#torch==2.5.1 textworld==1.6.1 fast_downward_textworld==20.6.2 
pip install alfworld==0.3.5 spacy==3.7.5
pip install -e .
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python-headless==4.5.5.64 fast-downward-textworld==20.6.1 
export ALFWORLD_DATA=~/.cache/alfworld
alfworld-download
