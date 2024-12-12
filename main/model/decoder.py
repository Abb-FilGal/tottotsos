import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        
        self.input_size = config["input_size"]
        self.lstm_dim = config["lstm_dim"]
        self.mel_dim = config["mel_dim"]
        self.max_decoder_steps = config["max_decoder_steps"]
        self.teacher_forcing_ratio = config["teacher_forcing_ratio"]
        
        self.lstm = nn.LSTM(self.input_size, self.lstm_dim, batch_first=True)
        self.output_layer = nn.Linear(self.lstm_dim, self.mel_dim)
        self.eos_token = torch.tensor([1.0] * self.mel_dim).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, encoder_output, hidden, target_mel=None):
        batch_size = encoder_output.size(0)
        
        # Ensure the encoder_output is of the right shape
        if encoder_output.size(-1) != self.input_size:
            raise RuntimeError(f"input.size(-1) must be equal to input_size. Expected {self.input_size}, got {encoder_output.size(-1)}")
        
        decoder_input = torch.zeros(batch_size, 1, self.mel_dim).to(encoder_output.device)
        decoder_output = []
        use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio
    
        hidden_state, cell_state = hidden

        for step in range(self.max_decoder_steps):
            if use_teacher_forcing and target_mel is not None and step < target_mel.size(1):
                decoder_input = target_mel[:, step].unsqueeze(1)
            else:
                if encoder_output.size(-1) != self.input_size:
                    encoder_output = encoder_output[:, :, :self.input_size]
                lstm_out, (hidden_state, cell_state) = self.lstm(decoder_input, (hidden_state, cell_state))
                mel_pred = self.output_layer(lstm_out)
                decoder_output.append(mel_pred)
                decoder_input = mel_pred

            if step > 0 and torch.allclose(decoder_input.squeeze(), self.eos_token, atol=1e-3):
                break
            
        decoder_output = torch.cat(decoder_output, dim=1)
    
        return decoder_output

