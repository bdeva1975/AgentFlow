# AgentFlow

An autonomous AI assistant built with Streamlit and OpenAI's API, capable of intelligent reasoning, planning, and executing tasks based on user input.

## Features

- **Autonomous Reasoning**: Analyzes queries, plans approaches, and executes multi-step tasks
- **File Analysis**: Processes various file types including PDFs, CSVs, Excel, and text files
- **Memory System**: Maintains conversation context with short and long-term memory
- **Self-Reflection**: Evaluates its own performance and improves over time
- **Interactive UI**: Clean Streamlit interface with adjustable agent settings

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/bdeva1975/agentflow.git
   cd agentflow
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

Run the application:
```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`

### Working with Files

1. Upload files using the "Upload Files" section
2. Ask questions about the files like "What insights can you get from the uploaded file?"
3. The AI will extract and analyze the content automatically

### Agent Settings

Customize the agent's behavior through the sidebar:
- **Model Selection**: Choose which OpenAI model to use
- **Autonomy Level**: Adjust how independent the agent is
- **Reasoning Display**: Toggle visibility of the agent's thinking process
- **Self-Reflection**: Enable/disable performance improvement

## Project Structure

- `app.py`: Main application code
- `requirements.txt`: List of dependencies
- `.env`: Environment variables (OpenAI API key)

## Dependencies

- Streamlit: Web interface
- OpenAI API: AI capabilities
- pandas: Data processing
- PyPDF2: PDF handling
- python-dotenv: Environment management

## License

[MIT](LICENSE)

## Acknowledgements

This project uses OpenAI's API for natural language processing capabilities. Built with ❤️ using Streamlit.
