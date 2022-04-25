seed = 5
utterance_max_length = 50
dialogue_max_length = 50
state_num = 6

PAD_ID = 0
UNK_ID = 1

epoch_num = 15
inference_step = 100
embedding_size = 300
hidden_size = 300
batch_size = 16
learning_rate = 1e-5
total_noise_ratio = 0.2
noise_ratio = 0.05

save_input_path = "./input_saving/"
log_path = "./log/"
output_path = "./output/"
save_model_path = "./saved_models/"
glove_path = "../glove/glove.840B.300d.txt"
data_path = "../dataset/"

NCE_weightage = 0.4
Prototype_weightage = 0.4
temperature=0.1
base_temperature=0.1
