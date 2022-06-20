seed = 5
utterance_max_length = 50
dialogue_max_length = 50

dataset = 'irc'

state_num = 16 if dataset == 'irc' else 6

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

exp_name = "run_1"

save_input_path = "./input_saving/"
log_path = "./log/{}/".format(exp_name)
output_path = "./output/{}/".format(exp_name)
save_model_path = "./saved_models/{}/".format(exp_name)
glove_path = "../glove/glove.840B.300d.txt"
data_path = "../dataset/"

lu_weightage = 0.4
ls_weightage = 0.4
temperature=0.1
base_temperature=0.1
