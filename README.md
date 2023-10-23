# Project Overview

The goal of this project was to build an entertaining and engaging generative chatbot that can carry out multi-turn conversations, adapt to context, and handle a variety of topics. To do this, GPT-2 small was fine tuned on a corpus of movie conversations from the [Cornell Movie Dialog dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). After training the model, chatbot architecture was built on top of it to allow the model to respond to user input and maintain context.


## Running the website locally
1. Clone the repo
2. Create a virtual environment
  ```sh
  python3 -m venv env
  source env/bin/activate
  ```
4. Install the required packages
  ```sh
  pip install -r "requirements.txt"
  ```
5. Navigate to the `/web_app` directory
6. Run the command
  ```sh
  flask run
  ```
6. Navigate to the localhost port running the website. You should see the website shown below.
7. Enjoy!

![image](https://github.com/julietlawton/generative-chat-bot/assets/124414966/d76cc7ab-d9a5-4244-8e72-7e6854be247a)
