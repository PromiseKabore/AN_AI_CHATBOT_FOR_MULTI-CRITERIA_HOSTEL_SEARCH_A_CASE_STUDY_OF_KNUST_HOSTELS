# KNUST Hostel AI Chatbot

AI chatbot for multi-criteria hostel search around KNUST. Interprets natural-language queries, filters hostels by price/distance/room type/amenities, and ranks results with TF-IDF + cosine similarity.

Based on the BSc thesis by Asford Okine and Promise Mawutor Wendpuoire Kabore (KNUST, 2026).

## Stack

- Python, Streamlit
- pandas
- Groq API (`openai/gpt-oss-120b`) for query interpretation
- scikit-learn (TF-IDF, cosine similarity) for ranking
- CSV dataset (`hostel_data.csv`), 2,728 listings / 1,000 hostels

## Setup

```bash
pip install -r requirements.txt
```

Create `.env`:

```
GROQ_API_KEY=your_key_here
```

## Run

```bash
streamlit run app.py
```

## Notes

- Dataset is a static dev/test set, partly synthetic.
- No booking/payment — info and recommendations only.
