# coding=utf-8
"""
@file    test_lstm
@date    2023/5/22 16:27
@author  zlf
"""
import torch
import torch.nn as nn
import torch.optim as optim


# Define the LSTM model for NER
class LSTMTagger(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        tag_space = self.fc(lstm_out)
        tag_scores = nn.functional.log_softmax(tag_space, dim=2)
        return tag_scores


if __name__ == '__main__':
    # Define the training data
    training_data = [
        ("The company is located in New York City", ["O", "O", "O", "O", "O", "B-LOC", "I-LOC", "I-LOC"]),
        ("She visited Paris last summer", ["O", "O", "B-LOC", "O", "B-TIME"]),
        # more training examples
    ]

    # Create vocabulary and label dictionaries
    word_to_idx = {}
    label_to_idx = {"O": 0, "B-LOC": 1, "I-LOC": 2, "B-TIME": 3, "I-TIME": 4}

    for sentence, labels in training_data:
        for word in sentence.split():
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)

    # Convert training data to numerical format
    numerical_data = []
    for sentence, labels in training_data:
        numerical_sentence = [word_to_idx[word] for word in sentence.split()]
        numerical_labels = [label_to_idx[label] for label in labels]
        numerical_data.append((numerical_sentence, numerical_labels))

    # Define hyperparameters
    input_dim = len(word_to_idx)
    hidden_dim = 10
    output_dim = len(label_to_idx)
    learning_rate = 0.01
    num_epochs = 10

    # Create the LSTM model
    model = LSTMTagger(input_dim, hidden_dim, output_dim)

    # Define the loss function and optimizer
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for inputs, targets in numerical_data:
            # Clear gradients
            model.zero_grad()

            # Convert inputs to tensors
            inputs = torch.tensor(inputs).unsqueeze(0)
            targets = torch.tensor(targets).unsqueeze(0)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = loss_function(outputs.view(-1, output_dim), targets.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()

    # Inference
    with torch.no_grad():
        model.eval()
        test_sentence = "New York City"
        numerical_test_sentence = [word_to_idx[word] for word in test_sentence.split()]
        inputs = torch.tensor(numerical_test_sentence).unsqueeze(0)
        outputs = model(inputs)
        predicted_labels = torch.argmax(outputs, dim=2).squeeze(0).tolist()

        predicted_tags = [list(label_to_idx.keys())[idx] for idx in predicted_labels]
        print("Test Sentence:", test_sentence)
        print("Predicted Tags:", predicted_tags)
