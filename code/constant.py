seed = 5
utterance_max_length = 50
dialogue_max_length = 50
state_num = 6

PAD_ID = 0
UNK_ID = 1

adopt_speaker = True

epoch_num = 7
inference_step = 1000
embedding_size = 300
hidden_size = 300

attention_size = 300 if not adopt_speaker else 300 + dialogue_max_length

batch_size = 16
learning_rate = 5e-5
total_noise_ratio = 0.2
noise_ratio = 0.05

exp_name = "run_28_entire_set_with_matching_lr_5e-5_epoch_7_mlp_supervised_k_label_adopt_speaker"

save_input_path = "./input_saving/"
log_path = "./log/{}/".format(exp_name)
output_path = "./output/{}/".format(exp_name)
save_model_path = "./saved_models/{}/".format(exp_name)
glove_path = "../glove/glove.840B.300d.txt"
data_path = "../dataset/"

NCE_weightage = 0.4
Prototype_weightage = 0.4
temperature=0.1
base_temperature=0.1
