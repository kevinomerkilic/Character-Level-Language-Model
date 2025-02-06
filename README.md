Character-Level Language Model - A Doll's House GPT

🚀 Welcome to the Character-Level Language Model! This project implements a Transformer-based character-level language model trained on Henrik Ibsen's A Doll’s House. The model learns to generate text that mimics the play’s structure, dialogue, and style.

📜 Project Overview

This model is built using PyTorch and follows the foundational principles of the GPT (Generative Pre-trained Transformer) architecture. It is trained on a dataset derived from A Doll's House, encoding text at a character level and generating new text in a similar style.

🎬 Watch the Full Tutorial Series on YouTube!
https://www.youtube.com/@Neural_and_Wires

This project is part of my YouTube series on Neural & Wires, where I explain step by step how to build this model from scratch!
📺 Watch Here: 🔗 Neural & Wires YouTube Channel

🛠 How It Works

Data Processing: Reads A Doll’s House and tokenizes it at the character level.
Model Training: A transformer model learns patterns in the text through self-attention and multi-head attention.
Text Generation: Once trained, the model can generate new lines of dialogue in the style of Henrik Ibsen’s play.
📝 Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/kevinomerkilic/character-level-language-model.git
cd character-level-language-model
2️⃣ Install Dependencies
Make sure you have Python and PyTorch installed:

pip install torch numpy tqdm
3️⃣ Run the Model
To train the model, run:

python train.py
To generate text after training:

python generate.py
📊 Training Performance

✅ Starting Loss: ~4.44
✅ Final Loss: ~1.26
✅ Results: Generates structured text resembling the original play, with room for improvement in coherence.

🔍 Future Improvements

Train on larger datasets (e.g., multiple Ibsen plays).
Implement better tokenization (BPE, word-level) for improved coherence.
Tune hyperparameters for better accuracy.
📢 Contribute & Stay Updated!

Feel free to fork this repository, contribute, and improve the model! If you enjoyed this project, subscribe to my YouTube channel for more tutorials on AI, machine learning, and deep learning! 🚀

📺 YouTube: 🔗 Neural & Wires https://www.youtube.com/@Neural_and_Wires
💬 IG: @omerionize
📧 Contact: oklc150@gmail.com
