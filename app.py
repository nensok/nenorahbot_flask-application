from flask import Flask, request, render_template
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

app = Flask(__name__)

# Load the model and tokenizer
mname = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotTokenizer.from_pretrained(mname)

# Create lists to keep track of previous user and bot responses
user_responses = []
bot_responses = []

# Define the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the chat page
@app.route('/chat', methods=['GET','POST'])
def chat():
    if request.method == 'POST':
        # Get user input
        user_input = request.form['user_input']

        # Encode user input and generate response
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        response_ids = model.generate(input_ids)
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Check if user input is empty and set default response
        if not user_input:
            response_text = "Hi I am Nenorahbot whats your name?"

        # Add user and bot responses to their respective lists
        user_responses.append(user_input)
        bot_responses.append(response_text)

    else:
        # Set default response for initial load of chat page
        response_text = "Hi I am Nenorahbot Whats your name?"

    # Render the chat page
    return render_template('index.html', response_text=response_text, user_responses=user_responses, bot_responses=bot_responses)


if __name__ == '__main__':
    app.run(debug=True)
