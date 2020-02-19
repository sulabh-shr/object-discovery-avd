from metric_learning_model import MetricLearningNet
from metric_learning_data import ActiveVisionTriplet
from loss import triplet_loss

import os
import json
import pickle
from tqdm import tqdm
from datetime import datetime
from time import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


def train(train_params, paths, model_params, data_params):
    epochs = train_params['epochs']
    batch_size = train_params['batch_size']
    val_size = train_params['val_size']
    learning_rate = train_params['learning_rate']

    dataset_root = paths['dataset']
    triplets_path = paths['triplets']
    output_root = paths['output_root']

    proposal_img_size = data_params['proposal_img_size']
    scene = data_params['scene']

    triplet_img_size = model_params['triplet_img_size']
    margin = model_params['margin']
    embedding_model = model_params['embedding_model']

    # OUTPUT FOLDER SETUP
    SLURM_SUFFIX = datetime.now().strftime("%d-%m-%Y_%H_%M_%S")

    try:
        SLURM_JOB_ID = str(os.environ["SLURM_JOB_ID"])
        SLURM_SUFFIX = SLURM_JOB_ID
    except KeyError:
        print('Slurm Job Id not avaialable')

    output_root = os.path.join(output_root, 'model_' + SLURM_SUFFIX)

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # DATASET SETUP
    dataset = ActiveVisionTriplet(dataset_root, triplets_path, instance=scene,
                                  image_size=proposal_img_size,
                                  triplet_image_size=triplet_img_size, get_labels=False,
                                  proposals_root=None, plot_original_proposals=False)
    num_val = round(len(dataset) * val_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [len(dataset) - num_val, num_val])

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # MODEL SETUP
    model = MetricLearningNet(model=embedding_model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = {'val_loss': [],
              'train_loss': []}

    with open(os.path.join(output_root, f'params.txt'), 'w') as f:
        _params = {
            'TRAIN_PARAMS': train_params,
            'PATHS': paths,
            'MODEL_PARAMS': model_params,
            'DATA_PARAMS': data_params
        }

        json.dump(_params, f)
        del _params

    with open(os.path.join(output_root, 'time.txt'), 'a') as f:
        f.write("START TIME: " + datetime.now().strftime("%m-%d-%Y_%H:%M:%S") + '\n')

    # TRAINING
    with tqdm(total=epochs) as epoch_bar:
        epoch_bar.set_description(f'Total Epoch Progress: '.ljust(20))

        for epoch in range(epochs):
            loss_per_iter = []

            with tqdm(total=len(train_data_loader)) as it_bar:
                it_bar.set_description(f'Epoch={epoch + 1}'.ljust(20))
                # start = time()

                for idx, (ref_list, pos_list, neg_list, labels) in enumerate(train_data_loader):
                    optimizer.zero_grad()
                    # print(f'Loading: {time()-start}')
                    # start = time()
                    ref_emb, pos_emb, neg_emb = model(ref_list, pos_list, neg_list)
                    # print(f'Forward: {time()-start}')

                    # start = time()
                    loss = triplet_loss(ref_emb, pos_emb, neg_emb, min_dist_neg=margin)
                    # print(f'Loss calc: {time()-start}')
                    loss_per_iter.append(loss.item())

                    # start = time()
                    loss.backward()
                    # print(f'Loss back: {time()-start}')

                    # start = time()
                    optimizer.step()
                    # print(f'Optimizer step: {time()-start}')

                    it_bar.set_postfix(train_loss=f'{loss:.4f}')
                    it_bar.update()

                    # start = time()

                losses['train_loss'].append(loss_per_iter)

                # VALIDATION of Epoch
                with torch.no_grad():
                    val_loss = 0
                    for ref_list, pos_list, neg_list, labels in val_data_loader:
                        ref_emb, pos_emb, neg_emb = model(ref_list, pos_list, neg_list)
                        val_loss += triplet_loss(ref_emb, pos_emb, neg_emb, min_dist_neg=margin)
                        break
                    val_loss = val_loss / len(val_dataset)

                losses['val_loss'].append(val_loss.item())

                it_bar.set_postfix(val_loss=f'{val_loss}')

            epoch_bar.update()

    torch.save(model.state_dict(), os.path.join(output_root, 'model_' +
                                                datetime.now().strftime("%m-%d-%Y_%H_%M_%S") +
                                                '.pth'))

    with open(os.path.join(output_root, 'losses.pickle'), 'wb') as f:
        pickle.dump(losses, f)

    with open(os.path.join(output_root, 'time.txt'), 'a') as f:
        f.write("END TIME: " + datetime.now().strftime("%m-%d-%Y_%H:%M:%S") + '\n')

    return model, losses


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory of the dataset.")
    parser.add_argument("--triplets_path", type=str, required=True,
                        help="Path to where triplets are stored.")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Path to save the model along with the filename.")
    parser.add_argument("--scene", type=str, required=True,
                        help="Scene for which the triplet is to be generated.")
    args = parser.parse_args()

    train_params = {
        'epochs': 5,
        'batch_size': 16,
        'val_size': 0.1,
        'learning_rate': 0.001,
    }

    paths = {
        'dataset': args.dataset_root,
        'triplets': args.triplets_path,
        'output_root': args.output_root,

    }

    data_params = {
        'proposal_img_size': (1920, 1080),
        'scene': args.scene
    }

    model_params = {
        'triplet_img_size': (224, 224),
        'margin': 1,
        'embedding_model': 'resent18'
    }

    train(train_params, paths, model_params, data_params)
