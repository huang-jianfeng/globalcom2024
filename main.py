
import os


# 'fedavg' 'streamfedavg' 'simple' 使用simple server client num必须是4，数据集为cifar10
# 'fedproxstream' 'fedprox'
# algorithm: 'fedproxstream'

if __name__ == '__main__':
    # os.system(f'python ./src/run.py --sample counter --unbalance 0.8 --agg avg --counter_type cmsketch')

    # for unbalance in [0.5,0.8,0.9]:
    #         os.system(f'python ./src/run.py --unbalance {unbalance }')
    #     os.system(f'python ./src/run.py --sample random --unbalance 1 --counter_type {type_c} --rand_seed {rand}')



    # for unbalance in [1]:
    # for rand_seed in [18390,239423,909]:
    # for rand_seed in [456,18390,909]:
        # for algorithm in ['motivation_run']:
            
        for algorithm in ['streamfedavg']:
        # #     os.system(f'python ./src/run.py  --algorithm {algorithm}   --sample counter --counter_type cmsketch' )
            # for sample in ['random','counter']:
            for sample in ['counter']:
        # #         # for local_lr in [0.01,0.02,0.03,0.04,0.05]:
        # #         for local_epoch in [1,2,3,4,5,6,7,8]:
        #         # for data_inc_num in [90,70,50,30,10]:
        #         # for sampling_ratio in [0.15,0.2]:
                if sample =='counter':
                    # for counter_type in ['array','cmsketch','bloomfilter']:
                    for counter_type in ['bloomfilter']:
                        os.system(f'python ./src/run.py  --algorithm streamfedavg   --sample {sample} --counter_type {counter_type} ' ) 
                else:
                        os.system(f'python ./src/run.py  --algorithm streamfedavg   --sample {sample}  ' ) 
                        # os.system(f'python ./src/run.py  --algorithm streamfedavg   --sample counter --counter_type bloomfilter ' ) 
                     
                # for counter_type in ['array','cmsketch']:
                # for unbalance in [0.75,0.80,0.85,0.90,0.95]:
                    # for sampling_ratio in['0.1']:
        # for dataset in ['mnist','cifar10','fmnist']:
                        # os.system(f'python ./src/run.py  --algorithm {algorithm}  --local_epoch {local_epoch}' )
    # os.system(f'python ./src/run.py  --algorithm streamfedavg   --rand_seed {213}  --sample random' )
    # os.system(f'python ./src/run.py  --algorithm streamfedavg   --rand_seed {4324}  --sample random' )
    # for rand_seed in  [45]:
    #  for unbalance in [0.8]:
    #     # for algorithm in ['powerd']:
    #     for algorithm in ['streamfedavg']:
    #         for sample in ['counter']:
    # #     # for unbalance in [0.8]:
    #             os.system(f'python ./src/run.py --unbalance {unbalance} --algorithm {algorithm} --rand_seed {rand_seed} --sample {sample}')
    # for rand_seed in  [32,45]:
    #  for unbalance in [0.8]:
    #     for algorithm in ['powerd']:
    # #     # for algorithm in ['streamfedavg']:
    # #         # for sample in ['random','counter']:
    # #     # for unbalance in [0.8]:
    #             os.system(f'python ./src/run.py --unbalance {unbalance} --algorithm {algorithm} --rand_seed {rand_seed}')
    # # # for unbalance in [0.93]:
    #         # os.system(f'python ./src/run.py --sample counter --unbalance {unbalance} --algorithm streamfedavg')

    #         rounds: 50
    #         seq_num: 1
    #  os.system(f'python ./src/run.py --alpha 0.1 --client_num 100')
    # for seq_num in [0,1,2,3,4,5]:
    #      os.system(f'python ./src/run.py --alpha {0.1} --client_num {4} --seq_num {seq_num}')