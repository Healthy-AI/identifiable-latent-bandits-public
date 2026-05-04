import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score

from models.sequence_regressor.model import SequenceModel, create_data_loader


def early_stopping(history, patience=5):
    if len(history['val_r2']) < patience or history['val_r2'][-1] < 0:
        return False
    recent_r2 = history['val_r2'][-patience:]
    if all(recent_r2[i] < recent_r2[i - 1] for i in range(1, patience)):
        return True
    return False


def train(x_time,
          treatment_id,
          rewards,
          n_treatment,
          lstm_layers,
          hidden_size,
          learning_rate,
          decay_factor,
          batch_size,
          num_epochs,
          train_dir,
          save_file='model.ckpt',
          save_steps=None,
          random_seed=None,
          val=None,
          n=0,l=0):
    """Sequence model training."""
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    save_file = os.path.join(train_dir, save_file)
    data_loader = create_data_loader(x_time, treatment_id, rewards, batch_size=batch_size, shuffle=True)
    model = SequenceModel(input_size=x_time.shape[-1], hidden_size=hidden_size, num_layers=lstm_layers, k_treatments=n_treatment)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_factor)
    #ema = torch.optim.swa_utils.AveragedModel(model)

    # Training loop
    history = {
        'loss': [],
        'loss0': [],
        'val_loss': [],
        'val_r2': [],
    }
    for epoch in range(num_epochs):

        # Validation step
        if val is not None:
            model.eval()
            with torch.no_grad():
                val_outputs = model(val['x_time'], val['treatment_id'])
                val_loss = criterion(val_outputs, val['rewards'])
                val_r2 = r2_score(val['treatment']['reward'].flatten(), val_outputs.cpu().numpy().flatten())
                history['val_loss'].append(val_loss.item())
                history['val_r2'].append(val_r2)
            print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}')
            print(f'Validation Loss: {val_loss.item():.4f}, R2: {val_r2:.4f}')

        model.train()
        for batch_x_time, batch_treatment_id, batch_rewards in data_loader:
            for time in range(1, batch_x_time.shape[1]+1):
                avg_loss = []
                outputs = model(batch_x_time[:, :time], batch_treatment_id[:, :time])
                loss = criterion(outputs, batch_rewards[:, :time])

                # Epoch 0
                if epoch == 0 and time == 1:
                    print(f'Epoch [{epoch}/{num_epochs}], Time step [{time}/{batch_x_time.shape[1]}], Loss: {loss.item():.4f}')
                    history['loss0'].append(loss.item())

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f'Epoch [{epoch+1}/{num_epochs}], Time step [{time}/{batch_x_time.shape[1]}], Loss: {loss.item():.4f}')
                history['loss0'].append(loss.item())
                avg_loss.append(loss.item())
        scheduler.step()
        history['loss'].append(loss.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if (epoch + 1) % save_steps == 0:
            torch.save(model.state_dict(), save_file)
            print(f'Model checkpoint saved at epoch {epoch+1}')

        patience = 5
        if early_stopping(history, patience=patience):
            print(f'Early stopping triggered due to validation R2 not increasing for {patience} epochs.')
            break

    # Save the final model
    model.history = history
    torch.save(model.state_dict(), save_file)
    print('Final model saved.')
    return history


def prepare_data(sensor, treatment_data, n_obs_per_seg, device='cpu'):
    time_steps = np.arange(0, sensor.shape[0], n_obs_per_seg)
    x_time = np.concatenate([np.expand_dims(sensor[i:i+n_obs_per_seg, :], 0) for i in time_steps], axis=0) # [Patient, time, features]
    x_time = torch.tensor(x_time, dtype=torch.float32).to(device)
    treatment_id = torch.tensor(treatment_data['treatment'], dtype=torch.int64).to(device)
    rewards = torch.tensor(treatment_data['reward'], dtype=torch.float32).to(device)
    return x_time, treatment_id, rewards


def sequence_model_wrapper(sensor, treatment_data, n_layers, n_obs_per_seg, num_epochs=10,
                            ckpt_dir='./', random_seed=None, val=None, **kwargs):
    # Training ----------------------------------------------------
    learning_rate = 0.01  # initial learning rate
    hidden_size = 64
    decay_factor = 0.1  # decay factor (tf.train.exponential_decay)
    batch_size = 512  # mini-batch size
    checkpoint_steps = 100  # interval to save checkpoint

    # Other -------------------------------------------------------
    train_dir = ckpt_dir  # save directory

    # Prepare Data and train
    train_dir = ckpt_dir  # save directory
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_time, treatment_id, rewards = prepare_data(sensor, treatment_data, n_obs_per_seg, device=device)
    if val is not None:
        x_test, treatment_data_test = val
        x_time_test, treatment_id_test, rewards_test = prepare_data(x_test, treatment_data_test, n_obs_per_seg, device=device)
        validation = {'x_time': x_time_test,
                      'treatment_id': treatment_id_test,
                      'rewards': rewards_test,
                      'treatment': treatment_data_test,
                      }
    n_treatment = torch.max(treatment_id).item()+1

    history = train(x_time=x_time,
                    treatment_id=treatment_id,
                    rewards=rewards,
                    n_treatment=n_treatment,
                    lstm_layers=n_layers//2,
                    hidden_size=hidden_size,
                    learning_rate=learning_rate,
                    decay_factor=decay_factor,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    train_dir=train_dir,
                    save_file='model.ckpt',
                    save_steps=checkpoint_steps,
                    random_seed=random_seed,
                    val=validation,
                    n=kwargs.get('n', 0),
                    l=kwargs.get('l', 0),
                    )
    return history


def load_model(ckpt_folder, data_dim, n_treatment, n_layers):
    model_path = os.path.join(ckpt_folder, 'model.ckpt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SequenceModel(input_size=data_dim, hidden_size=64, num_layers=n_layers//2, k_treatments=n_treatment)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model
