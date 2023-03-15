# AI lotto number guesser

## Description
This is an AI lotto number guesser. It is trained on the last 6000 draws of the German lotto. It uses a neural network to guess the next draw.

## Disclaimer
This is just a fun project. I know it is not possible to predict the next lotto numbers. I am not responsible for any losses you might have.

# How to use
## Requirements
- Python 3.6
- Tensorflow 1.4.0
- Pandas 0.22.0
- requests 2.18.4

## Usage
1. Clone the repository
2. Install the requirements
3. Runs the web scraper with `python3 web_scraper.py`
4. Train the neural network with `python3 train.py`
5. Run the lotto guesser with `python3 lotto.py`