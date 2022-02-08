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
python main.py --name CTHFSGM \
               --model cthfsgm \
               --augment \
               --dataset $DATASET \
               --likelihood $LL \
               --hidden-dim $HD \
               --c-dim 32 \
               --z-dim 32 \
               --output-dir /output \
               --alpha-step 0.98 \
               --alpha 2 \
               --adjust-lr \
               --scheduler plateau \
               --sample-size $SS \
               --sample-size-test $SS \
               --num-classes 1 \
               --learning-rate 1e-3 \
               --epochs $EP \
               --batch-size $BS \
               --tag $TAG
}

# Run config
case $RUN in
   celeba_binary)
      run celeba binary 5 1000 256 50 cthfsgm_celeba
      ;;
   celeba_mixlog)
      run celeba discretized_mix_logistic 5 1000 256 50 cthfsgm_celeba
      ;;
   omniglot)
      run omniglot_back_eval binary 5 1000 128 100 cthfsgm_omniglot
      ;;
   omniglot_random)
      run omniglot_random binary 5 1000 128 100 cns_omniglot_r
      ;;
   omniglot_ns)
      run omniglot_ns binary 5 1000 128 100 cthfsgm_omniglot_ns
      ;;
esac
       