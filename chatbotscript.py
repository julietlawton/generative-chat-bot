# %%
import os
import chardet
import re
import ast
import pickle
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pandarallel import pandarallel
from torch.utils.data import Dataset, DataLoader
import torch

pd.set_option("display.max_colwidth", None)
pandarallel.initialize(progress_bar=True, nb_workers=8)

# %%
DATASET_FILENAME = "movie_conversations.pkl"
MAX_TOKEN_LENGTH = 1024
dataset_generated = False

if os.path.exists(DATASET_FILENAME):
    dataset_generated = True

# %%
if not dataset_generated:
    # Detect encoding of the movie lines file
    cwd = os.getcwd()

    with open(cwd + "/movie_lines/movie_lines.txt", "rb") as f:
        result = chardet.detect(f.read())
        movie_lines_encoding = result["encoding"]

    with open(cwd + "/movie_lines/movie_conversations.txt", "rb") as f:
        result = chardet.detect(f.read())
        movie_conversations_encoding = result["encoding"]

    print(movie_lines_encoding)
    print(movie_conversations_encoding)
    # movie_lines_encoding = "Windows-1252"
    # movie_conversations_encoding = "ascii"

# %%
if not dataset_generated:
    # Collect individual movie lines
    with open(cwd + "/movie_lines/movie_lines.txt", "r", encoding=movie_lines_encoding) as f:
        content = f.read()

    lines = content.split("\n")

    # Print first 5 lines and verify length is correct
    print(lines[:5])

    # Remove last element of lines because its an empty string
    lines = lines[:-1]
    print(len(lines))

# %%
if not dataset_generated:
    # Initialize containers for values to put in dataframe
    line_numbers_dict = {}

    for line in lines:
        # Split on whitespace
        split = line.split(" ")
        line_number = split[0]
        character_id = split[2]

        # Extract the text after the last "+" character
        l = re.split(r'\+\s+(?=[^+]*$)', line)[-1]
        line_numbers_dict[line_number] = (character_id, l)


    # Create dataframe from extracted values
    print(dict(itertools.islice(line_numbers_dict.items(), 10)))

# %%
if not dataset_generated:
    # Collect movie conversation lists
    with open(cwd + "/movie_lines/movie_conversations.txt", "r", encoding=movie_conversations_encoding) as f:
        content = f.read()

    lines = content.split("\n")

    # Print first 5 lines and verify length is correct
    print(lines[:5])

    # Remove last element of lines because its an empty string
    lines = lines[:-1]
    print(len(lines))

# %%
if not dataset_generated:
    # Initialize containers for values to put in dataframe
    speaker1_ids = []
    speaker2_ids = []
    conversation_lines = []

    for line in lines:
        # Split on whitespace
        split = line.split(" ")
        speaker1_ids.append(split[0])
        speaker2_ids.append(split[2])

        # Extract the text after the last "+" character
        l = re.split(r'\+\s+(?=[^+]*$)', line)[-1]
        l = ast.literal_eval(l)
        conversation_lines.append(l)


    # Create dataframe from extracted values
    movie_conversations = pd.DataFrame(list(zip(speaker1_ids, speaker2_ids, conversation_lines)), columns=["speaker1_id", "speaker2_id", "conversation_lines"])
    print(movie_conversations.head())
    movie_conversations.info()

# %%
# Function for turning movie lines into multi-turn conversations for training
def create_conversation_turns(row):
    speaker1 = row["speaker1_id"]
    conversation_list = row["conversation_lines"]
    convo = ""
    # For each line, add it to the conversation with a role label
    for line_id in conversation_list:
        movie_line = line_numbers_dict[line_id]
        text = movie_line[1]

        if movie_line[0] == speaker1:
            convo += f"<USER>: {text} \n"
        else:
            convo += f"<AGENT>: {text} \n"
        
    return convo

# %%
if not dataset_generated:
    # Try it on a sample for testing
    sample = movie_conversations.copy().iloc[:2]
    sample["conversation"] = sample.apply(create_conversation_turns, axis=1)
    print(sample)

    # Call create_conversation_turns on every row of the dataframe
    movie_conversations["conversation"] = movie_conversations.apply(create_conversation_turns, axis=1)

    # Drop speaker id columns because they aren't needed anymore
    movie_conversations.drop(columns=["speaker1_id", "speaker2_id"], inplace=True)

# %%
if not dataset_generated:
    print(movie_conversations.head())
    print(movie_conversations["conversation"].head(1).values[0])

# %%
if not dataset_generated:
    # Get the token count for each conversation so we can split the ones that are too long
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    movie_conversations["token_count"] = movie_conversations["conversation"].apply(lambda x: len(tokenizer.encode(x)))
    movie_conversations.head()

# %%
if not dataset_generated:    
    # Get the conversation entries that are over the token limit
    over_token_limit = movie_conversations[movie_conversations["token_count"] > MAX_TOKEN_LENGTH].index
    print(over_token_limit)

# %%
if not dataset_generated:
    # NOTE: Be careful not to run this more than once without reseting the dataframe
    new_entries = []

    # For each conversation that is too long, split it in half
    for idx in over_token_limit:
        lines_to_split = movie_conversations.iloc[idx]["conversation_lines"]
        split_idx = (len(lines_to_split)//2)
        first_half = lines_to_split[:split_idx]
        second_half = lines_to_split[split_idx:]

        first_convo = ""
        second_convo = ""
        # For each line, add it to the conversation
        for i, line_id in enumerate(first_half):
            movie_line = line_numbers_dict[line_id]
            text = movie_line[1]

            if i%2 == 0:
                first_convo += f"<USER>: {text} \n"
            else:
                first_convo += f"<AGENT>: {text} \n"

        new_entries.append({"conversation_lines": first_half, "conversation": first_convo})

        for i, line_id in enumerate(second_half):
            movie_line = line_numbers_dict[line_id]
            text = movie_line[1]

            if i%2 == 0:
                second_convo += f"<USER>: {text} \n"
            else:
                second_convo += f"<AGENT>: {text} \n"

        new_entries.append({"conversation_lines": second_half, "conversation": second_convo})

    # Add the new entries from splitting and drop the originals
    movie_conversations = pd.concat([movie_conversations, pd.DataFrame(new_entries)], axis=0, ignore_index=True)
    movie_conversations.drop(index=over_token_limit, inplace=True)

    movie_conversations.reset_index(inplace=True, drop=True)
    movie_conversations.info()

# %%
if not dataset_generated:
    # Check that no conversations are over the token limit
    print(movie_conversations[movie_conversations["token_count"] > MAX_TOKEN_LENGTH])

# %%
if not dataset_generated:
    with open(DATASET_FILENAME, "wb") as f:
        print("Writing dataframe to file")
        pickle.dump(movie_conversations, f)
else:
    with open(DATASET_FILENAME, "rb") as f:
        print("Loading dataframe from file")
        movie_conversations = pickle.load(f)

print(movie_conversations.head())
print(movie_conversations["conversation"].head(1).values[0])

# %%
model_name = "gpt2-medium"  # You can specify the desired model size
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# %%
tokenized_texts = [tokenizer.encode(conversation, return_tensors="pt") for conversation in movie_conversations["conversation"]]

# %%

batch_size = 4
learning_rate = 5e-5
num_epochs = 1
class ConversationDataset(Dataset):
    def __init__(self, tokenized_texts, tokenizer):
        self.tokenized_texts = tokenized_texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        return self.tokenized_texts[idx]

dataset = ConversationDataset(tokenized_texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch in dataloader:
        inputs = batch.to(device)
        labels = inputs.clone()

        optimizer.zero_grad()

        outputs = model(inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# %%
def generate_response(prompt, model, tokenizer, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage
prompt = "User: Hello, how are you?"
response = generate_response(prompt, model, tokenizer)
print(response)


