# 两个任务间的loss比例系数实验
exp_name=alpha_exp_new_restaurant

for alpha in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
  # 下面是最好的参数设定，在restaurant数据集上，acc：86.88
  python run.py --gat_bert \
    --embedding_type bert \
    --output_dir data/output-gcn \
    --dropout 0.2 \
    --hidden_size 200 \
    --learning_rate 5e-5 \
    --exp_name $exp_name/alpha_$alpha \
    --per_gpu_train_batch_size 16 \
    --use_bert_global t \
    --use_gat_feature t \
    --use_ner_feature t \
    --use_cross_attn t \
    --save_model f \
    --alpha $alpha

  #R-GAT+BERT in laptop
#  python run.py --gat_bert \
#    --embedding_type bert \
#    --dataset_name laptop \
#    --output_dir data/output-gcn-laptop \
#    --dropout 0.3 \
#    --num_heads 7 \
#    --hidden_size 200 \
#    --learning_rate 5e-5 \
#    --exp_name $exp_name/alpha_$alpha \
#    --per_gpu_train_batch_size 16 \
#    --use_bert_global t \
#    --use_gat_feature t \
#    --use_ner_feature t \
#    --use_cross_attn t \
#    --save_model f \
#    --alpha $alpha
#
#  #R-GAT+BERT in twitter
#  python run.py --gat_bert \
#    --embedding_type bert \
#    --dataset_name twitter \
#    --output_dir data/output-gcn-twitter \
#    --dropout 0.2  \
#    --hidden_size 200 \
#    --learning_rate 5e-5 \
#    --exp_name $exp_name/alpha_$alpha \
#    --per_gpu_train_batch_size 16 \
#    --use_bert_global t \
#    --use_gat_feature t \
#    --use_ner_feature t \
#    --use_cross_attn t \
#    --save_model f \
#    --alpha $alpha
done
#python run.py --gat_our --highway --num_heads 7 --dropout 0.8 # R-GAT in restaurant
#python run.py --gat_our --dataset_name laptop --output_dir data/output-gcn-laptop --highway --num_heads 9 --per_gpu_train_batch_size 32 --dropout 0.7 --num_layers 3 --hidden_size 400 --final_hidden_size 400 # R-GAT in laptop
#python run.py --gat_our --dataset_name twitter --output_dir data/output-gcn-twitter --highway --num_heads 9 --per_gpu_train_batch_size 8 --dropout 0.6 --num_mlps 1 --final_hidden_size 400 # R-GAT in laptop