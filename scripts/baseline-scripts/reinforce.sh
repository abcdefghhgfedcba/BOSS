# set up machine parameters
NUM_CPUS=5
NUM_GPUS=1
for TASK_NAME in \
    ant \
    chembl \
    dkitty \
    superconductor \
    tf-bind-8 \
    tf-bind-10  ; do

  for ALGORITHM_NAME in \
      reinforce ; do
  
    # launch several model-based optimization algorithms using the command line interface
    # for example: 
    # (design-baselines) name@computer:~/$ cbas gfp \
    #                                        --local-dir ~/results/cbas-gfp \
    #                                        --cpus 32 \
    #                                        --gpus 8 \
    #                                        --num-parallel 8 \
    #                                        --num-samples 8
    CUDA_VISIBLE_DEVICES=1 $ALGORITHM_NAME $TASK_NAME \
      --local-dir results/baselines/$ALGORITHM_NAME-$TASK_NAME \
      --cpus $NUM_CPUS \
      --gpus $NUM_GPUS \
      --num-parallel 1 \
      --num-samples 8
    
  done
  
done

# design-baselines make-table --dir results/baselines/$ALGORITHM_NAME-$TASK_NAME --group A --percentile 100th


