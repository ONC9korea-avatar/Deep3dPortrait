conda activate deep3dportrait

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/

python step1_recon_3d_face.py
# read
python step2_face_segmentation.py
# read
python step3_get_head_geometry.py
# read
python step4_save_obj.py

unset LD_LIBRARY_PATH

conda deactivate