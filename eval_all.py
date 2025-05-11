import os
configs_list = [
    './configs/cfg_ade20k.py',
    './configs/cfg_city_scapes.py',
    './configs/cfg_coco_object.py',
    './configs/cfg_coco_stuff164k.py',
    './configs/cfg_context59.py',
    './configs/cfg_context60.py',
    './configs/cfg_voc20.py',
    './configs/cfg_voc21.py'
]

for config in configs_list:
    config_name = os.path.basename(config).replace('cfg_', '').replace('.py', '')
    work_dir = f"logs/{config_name}"
    print(f"Running {config}")
    os.system(f"python eval.py --config {config} --work-dir {work_dir}")
