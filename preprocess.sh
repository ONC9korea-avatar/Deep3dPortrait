data_path=/workspace/Deep3dPortrait/input/doo

conda activate face_alignment

python scripts/landmark.py $data_path # generate facial landmarks

# generate face parsing
cd third_party/face-parsing.PyTorch
python test.py $data_path
cd -

# convert face parsing labels
python scripts/convert_mask.py $data_path/mask

conda deactivate