conda activate deep3dportrait

export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/

python step1_recon_3d_face.py /workspace/Deep3dPortrait/input/doo
python step2_face_segmentation.py
python step3_get_head_geometry.py
python step4_save_obj.py

unset LD_LIBRARY_PATH

conda deactivate