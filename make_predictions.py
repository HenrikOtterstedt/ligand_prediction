import torch
import argparse
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def load_embeddings(filepath):
    embeddings = {}
    with h5py.File(filepath, 'r') as f:
        for new_identifier in f.keys():
            embeddings[f[new_identifier].attrs["original_id"]] = np.array(f[new_identifier])
    return embeddings


def combination_approach(embeddings, model_dir, output_dir):
    # load the models and put them into evaluation mode
    m_model = torch.load(model_dir+"/combination_m.pth")
    m_model.eval().to(device)
    n_model = torch.load(model_dir+"/combination_n.pth")
    n_model.eval().to(device)
    s_model = torch.load(model_dir+"/combination_s.pth")
    s_model.eval().to(device)

    # prepare the output file
    with open(output_dir + "/combination_predictions.tsv", "a+") as file:
        # check if the file es empty, if so write the header
        file.seek(0)
        if not len(file.read(100)) > 0:
            file.write("protein_id\tm_probability\tn_probability\ts_probability\n")

        # start the prediction process
        with torch.no_grad():
            for prot_id in embeddings:
                embedding = torch.from_numpy(np.array(embeddings[prot_id])).to(device)
                embedding = embedding.type(torch.float32)

                m_prediction = m_model(embedding)
                n_prediction = n_model(embedding)
                s_prediction = s_model(embedding)

                # apply softmax function
                m_prediction = F.softmax(m_prediction, dim=0)
                n_prediction = F.softmax(n_prediction, dim=0)
                s_prediction = F.softmax(s_prediction, dim=0)

                # write the output
                line = f"{prot_id}\t{m_prediction[1].item()}\t{n_prediction[1].item()}\t{s_prediction[1].item()}\n"
                file.write(line)


def three_class_predictor(embeddings, model_dir, output_dir):
    # load the model
    model = torch.load(model_dir + "/3_class_predictor.pth")
    model.eval().to(device)

    # prepare the output file
    with open(output_dir + "/3_class_predictor_predictions.tsv", "a+") as file:
        # check if the file es empty, if so write the header
        file.seek(0)
        if not len(file.read(100)) > 0:
            file.write("protein_id\tm_probability\tn_probability\ts_probability\n")

        # start the prediction process
        with torch.no_grad():
            for prot_id in embeddings:
                embedding = torch.from_numpy(np.array(embeddings[prot_id])).to(device)
                embedding = embedding.type(torch.float32)

                prediction = model(embedding)
                # apply sigmoid function
                prediction = torch.sigmoid(prediction)

                # write the output
                line = f"{prot_id}\t{prediction[0].item()}\t{prediction[1].item()}\t{prediction[2].item()}\n"
                file.write(line)


def seven_class_predictions(embeddings, model_dir, output_dir):
    # load the model
    model = torch.load(model_dir + "/7_class_predictor.pth")
    model.eval().to(device)

    # prepare the output file
    with open(output_dir + "/7_class_predictor_predictions.tsv", "a+") as file:
        # check if the file es empty, if so write the header
        file.seek(0)
        if not len(file.read(100)) > 0:
            file.write("protein_id\tm_probability\tn_probability\ts_probability\tmn_probability\tms_probability\t"
                       "ns_probability\tmns_probability\n")

        # start the prediction process
        with torch.no_grad():
            for prot_id in embeddings:
                embedding = torch.from_numpy(np.array(embeddings[prot_id])).to(device)
                embedding = embedding.type(torch.float32)

                prediction = model(embedding)
                # apply softmax function
                prediction = F.softmax(prediction, dim=0)

                # write the output
                line = f"{prot_id}\t{prediction[0].item()}\t{prediction[1].item()}\t{prediction[2].item()}\t" \
                       f"{prediction[3].item()}\t{prediction[4].item()}\t{prediction[5].item()}\t{prediction[6].item()}\n"
                file.write(line)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate_input=0.5, dropout_rate_hidden=0.3):
        super(NeuralNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate_input),   # input layer dropout
            nn.Linear(input_size, hidden_size),
            nn.Dropout(dropout_rate_hidden),   # hidden layer dropout
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        out = self.classifier(x)
        # put activation function and softmax here if not using CrossEntropyLoss
        return out


if __name__ == '__main__':
    # find the appropriate device to run the code on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using Device {device}")

    # parse the arguments
    parser = argparse.ArgumentParser(description="Predict ligand binding for all proteins that have an embedding in the embeddings file.")
    parser.add_argument("-e", "--embeddings", help="Filepath to embeddings file", required=True, nargs=1)
    parser.add_argument("-o", "--output_dir", help="Filepath to the output directory", required=True, nargs=1)
    parser.add_argument("-m", "--models", help="Filepath to the models directory", required=True, nargs=1)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", action="store_true", help="Indicates the use of the 3-class Predictor.")
    group.add_argument("-s", action="store_true", help="Indicates the use of the 7-class Predictor.")
    group.add_argument("-c", action="store_true", help="Indicated the use of the combination approach.")

    args = parser.parse_args()

    # load the embeddings
    embeddings = load_embeddings(args.embeddings[0])

    # check which of the models should be used
    if args.c:
        combination_approach(embeddings, args.models[0], args.output_dir[0])
    elif args.t:
        three_class_predictor(embeddings, args.models[0], args.output_dir[0])
    elif args.s:
        seven_class_predictions(embeddings, args.models[0], args.output_dir[0])
