# Energy Profiling Thesis Pipeline

Geospatial energy profiling pipeline for master's thesis.
Extracts 38 energy-relevant features from 7 open data sources
for 1000 globally stratified coordinates, converts to text labels,
and generates energy profile captions via Grok + LangChain.

## Setup
1. Python 3.11 required
2. Create virtual environment: `python3.11 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install -r requirements.txt`
5. Copy `.env` and fill in API keys