#python3 build_chunks.py
python3 t10nrun_1_cora_sage.py --host_file hosts_docker_1_3 --tag_id t10nresampling_cora_sage --task train --train_num_proc 1
python3 t10nrun_1_ogbar_sage.py --host_file hosts_docker_1_3 --tag_id t10nresampling_ogbar_sage --task train --train_num_proc 1
