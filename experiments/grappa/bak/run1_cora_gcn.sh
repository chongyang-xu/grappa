#python3 build_chunks.py
python3 t10nrun_1_cora_gcn.py --host_file hosts_docker_1_2 --tag_id t10nresampling_cora_gcn --task train --train_num_proc 1
python3 t10nrun_1_ogbar_gcn.py --host_file hosts_docker_1_2 --tag_id t10nresampling_ogbar_gcn --task train --train_num_proc 1
