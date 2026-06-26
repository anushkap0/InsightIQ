# InsightIQ

InsightIQ is a Streamlit-based AI business analyzer that helps you upload review data and extract sentiment insights using Hugging Face NLP models. It is built for fast, interactive review analysis and is designed to showcase AI/ML and web development skills.

## Features

- Upload CSV files containing customer reviews.
- Analyze sentiment with a Hugging Face transformer model.
- Process reviews in batches for better performance.
- Show sentiment labels and confidence scores.
- Built with Streamlit for a clean and simple UI.
- Uses caching to reduce repeated load time.

## Demo

Add your live app link here:

```text
https://your-app-link.streamlit.app
```

## Tech Stack

- Python
- Streamlit
- Pandas
- PyTorch
- Hugging Face Transformers
- python-dotenv

## Project Structure

```text
InsightIQ/
│
├── app.py
├── sentiment_analyzer.py
├── trend_detector.py
├── rag_chatbot.py
├── requirements.txt
├── .env
└── README.md
```

## How It Works

1. The user uploads a CSV file with review text.
2. The app checks whether the required text column exists.
3. Reviews are analyzed using a sentiment model.
4. The app returns labels and confidence values.
5. The output can be used to understand customer feedback and business trends.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/InsightIQ.git
cd InsightIQ
```

Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
HUGGINGFACE_API_TOKEN=your_token_here
```

## Run the App

```bash
streamlit run app.py
```

## Sample CSV Format

Your CSV should contain a review text column like this:

```csv
review_text
"This product is amazing!"
"The service was slow and unhelpful."
```

## Features in Detail

- **Batch processing:** helps handle larger review datasets faster.
- **Caching:** avoids reloading the model every time.
- **Confidence scores:** shows how certain the model is about each prediction.
- **Interactive interface:** makes the app easy to use for non-technical users.

## Future Improvements

- Add sentiment trend visualizations.
- Add export options for analysis results.
- Add category-level business insights.
- Support more languages.
- Improve dashboard styling and layout.

## Screenshots

Add screenshots of the app here.

## Contributing

Contributions are welcome. For major changes, please open an issue first to discuss what you would like to improve.

## License

Add your preferred license here.

## Acknowledgements

- Streamlit
- Hugging Face Transformers
- PyTorch

