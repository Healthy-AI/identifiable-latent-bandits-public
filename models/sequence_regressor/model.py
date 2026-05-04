import torch
import torch.nn as nn
import torch.utils.data as data


class SequentialDataset(data.Dataset):

    def __init__(self, x_time, treatment_id, rewards):
        self.x_time = x_time
        self.treatment_id = treatment_id
        self.rewards = rewards

    def __len__(self):
        return len(self.x_time)

    def __getitem__(self, idx):
        return self.x_time[idx], self.treatment_id[idx], self.rewards[idx]


def create_data_loader(x_time, treatment_id, rewards, batch_size=32, shuffle=True):
    dataset = SequentialDataset(x_time, treatment_id, rewards)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class SequenceModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, k_treatments):
        super(SequenceModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.k_treatments = k_treatments
        self.encoder_lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder_lstm = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.treatment_matrix = nn.Linear(hidden_size, k_treatments)

    def forward(self, x_time, treatment_id):
        encoder_output, hidden_e = self.encoder_lstm(x_time)
        decoder_output, hidden_d = self.decoder_lstm(encoder_output)
        potential_outcomes = self.treatment_matrix(decoder_output.squeeze(1))
        if potential_outcomes.ndim == 2:
            potential_outcomes = potential_outcomes.unsqueeze(1)
        reward = torch.gather(potential_outcomes, -1, treatment_id.unsqueeze(-1)).squeeze()
        reward = reward if reward.ndim == 2 else reward.unsqueeze(1)
        return reward

    def get_potential_outcomes(self, x_time):
        """Only used in evaluation."""
        x_time = torch.as_tensor(x_time, dtype=torch.float32)
        encoder_output, hidden_e = self.encoder_lstm(x_time)
        decoder_output, hidden_d = self.decoder_lstm(encoder_output)
        potential_outcomes = self.treatment_matrix(decoder_output.squeeze(1))
        if potential_outcomes.ndim == 2:
            potential_outcomes = potential_outcomes.unsqueeze(1)
        return potential_outcomes


class MeanPredictorModel(nn.Module):
    """Dummy Model mimicking the SequenceModel for evaluation purposes."""

    def __init__(self, input_size, hidden_size, num_layers, k_treatments):
        super(MeanPredictorModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.k_treatments = k_treatments
        self.predictor = nn.Linear(1, k_treatments, bias=True)

    def forward(self, x_time, treatment_id):
        dummy_input = torch.ones((x_time.shape[0], x_time.shape[1], 1))
        potential_outcomes = self.predictor(dummy_input)
        if potential_outcomes.ndim == 2:
            potential_outcomes = potential_outcomes.unsqueeze(1)
        reward = torch.gather(potential_outcomes, -1, treatment_id.unsqueeze(-1)).squeeze()
        reward = reward if reward.ndim == 2 else reward.unsqueeze(1)
        return reward

    def get_potential_outcomes(self, x_time):
        """Only used in evaluation."""
        dummy_input = torch.ones((x_time.shape[0], x_time.shape[1], 1))
        potential_outcomes = self.predictor(dummy_input)
        if potential_outcomes.ndim == 2:
            potential_outcomes = potential_outcomes.unsqueeze(1)
        return potential_outcomes
