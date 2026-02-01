#python3 build_chunks.py
python3 t10nrun_1_cora_gat.py --host_file hosts_docker_1_1 --tag_id t10nresampling_cora_gat --task train --train_num_proc 1
python3 t10nrun_1_ogbar_gat.py --host_file hosts_docker_1_1 --tag_id t10nresampling_ogbar_gat --task train --train_num_proc 1
