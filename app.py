from flask import Flask, render_template, request, jsonify
from ai_agent import AIAgent

app = Flask(__name__)
agent = AIAgent()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    
    if message.lower() in ['exit', 'quit']:
        return jsonify({'response': 'Bye! 👋'})
    
    try:
        response = agent.process_message(message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)