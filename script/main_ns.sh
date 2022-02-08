#! /bin/sh
GPU=$1
RUN=$2

run()
{
DATASET=$1
LL=$2
SS=$3
EP=$4
HD=$5
BS=$6
TAG=$7

CUDA_VISIBLE_DEVICES=$GPU \

python main.py --name NS \
               --model ns \
               --augment \
               --dataset $DATASET \
               --likelihood $LL \
               --hidden-dim $HD \
               --c-dim 32 \
               --z-dim 32 \
               --output-dir /output \
               --alpha-step 0.98 \
               --alpha 2 \
               --sample-size $SS \
               --sample-size-test $SS \
               --num-classes 1 \
               --adjust-lr \
               --scheduler plateau \
               --epochs $EP \
               --batch-size $BS \
               --tag $TAG
}

# Run config
case $RUN in
   omniglot)
      run omniglot_back_eval binary 5 1000 512 100 ns_omniglot
      ;;
   omniglot_random)
      run omniglot_random binary 5 1000 512 100 ns_omniglot_r
      ;;
   omniglot_ns)
      run omniglot_ns binary 5 1000 512 100 ns_omniglot_ns
      ;;
esac
