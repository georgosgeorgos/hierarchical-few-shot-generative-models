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
LR=$7
TAG=$8

CUDA_VISIBLE_DEVICES=$GPU \
python main.py --name CHFSGM_MULTISCALE \
               --model chfsgm_multiscale \
               --augment \
               --dataset $DATASET \
               --str-enc 32-9,16-6,8-6,4-6,2-3,1-3 \
               --str-gen-z 8-6,4-6,2-3,1-3 \
               --str-gen-c 8-6,4-6,2-3,1-3 \
               --str-dec 32-9,16-6,8-6 \
               --likelihood $LL \
               --aggregation-mode mean \
               --hidden-dim $HD \
               --c-dim 32 \
               --z-dim 32 \
               --output-dir /output \
               --alpha-step 0.9 \
               --alpha 2 \
               --adjust-lr \
               --scheduler plateau \
               --sample-size $SS \
               --sample-size-test $SS \
               --num-classes 1 \
               --learning-rate $LR \
               --epochs $EP \
               --batch-size $BS \
               --tag $TAG
}

# Run config
case $RUN in
   cifar100_binary)
      run cifar100 binary 5 1000 128 32 2e-4 chfsgm_multi_cifar100
      ;;
   cifar100_mixlog)
      run cifar100 discretized_mix_logistic 5 1000 128 32 2e-4 chfsgm_multi_cifar100
      ;;
   celeba_binary)
      run celeba binary 5 1000 128 16 2e-4 chfsgm_multi_celeba
      ;;
   celeba_mixlog)
      run celeba discretized_mix_logistic 5 1000 128 16 2e-4 chfsgm_multi_celeba
      ;;
   omniglot)
      run omniglot_back_eval binary 5 1000 128 16 2e-4 chfsgm_multi_omniglot
      ;;
   omniglot_ns)
      run omniglot_ns binary 5 1000 128 16 2e-4 chfsgm_multi_omniglot_ns
      ;;
   omniglot_ns_large)
      run omniglot_ns binary 5 1000 256 100 2e-4 chfsgm_multi_omniglot_ns_large
      ;;
   celeba_mixlog_large)
      run celeba discretized_mix_logistic 5 1000 256 50 2e-4 chfsgm_multi_celeba_large
      ;;
esac
       