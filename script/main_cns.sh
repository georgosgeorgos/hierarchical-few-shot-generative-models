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

python main.py --name CNS \
               --model cns \
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
               --learning-rate 2e-4 \
               --epochs $EP \
               --batch-size $BS \
               --tag $TAG
}

# Run config
case $RUN in
   celeba_binary)
      run celeba binary 5 1000 128 50 cns_celeba
      ;;
   celeba_mixlog)
      run celeba discretized_mix_logistic 5 1000 256 50 cns_celeba
      ;;
   omniglot)
      run omniglot_back_eval binary 5 1000 256 100 cns_omniglot
      ;;
   omniglot_ns)
      run omniglot_ns binary 5 1000 256 100 cns_omniglot_ns
      ;;
   omniglot_ns2)
      run omniglot_ns binary 2 1000 256 100 cns_omniglot_ns2
      ;;
   omniglot_ns10)
      run omniglot_ns binary 10 1000 256 100 cns_omniglot_ns10
      ;;
esac
