import bs4.element
import requests
from bs4 import BeautifulSoup
import pandas as pd


def scraper() -> None:
    all_numbers: list[list[str, int]] = []
    years: list = [i for i in range(1955, 2023 + 1)]
    for number in years:
        print("Scraping year: " + str(number))
        response: requests.Response = requests.get(f'https://www.lottozahlenonline.de/statistik/beide-spieltage/lottozahlen-archiv.php?j={number}#lottozahlen-archive')
        soup: BeautifulSoup = BeautifulSoup(response.text, 'html.parser')

        all_drawings: bs4.element.ResultSet = soup.find_all('div', class_='zahlensuche_rahmen')

        for drawing in all_drawings:
            date: str = drawing.find('time', class_='zahlensuche_datum').text.strip()
            numbers: bs4.element.ResultSet = drawing.find_all('div', class_='zahlensuche_zahl')
            numbers: list[int] = [int(number.text) for number in numbers]
            super_number: str or int = drawing.find('div', class_='zahlensuche_zz').text
            if super_number == '':
                super_number = -1
            data: list = [date, *numbers, super_number]
            all_numbers.append(data)

    df: pd.DataFrame = pd.DataFrame(all_numbers, columns=['date', 'number_1', 'number_2', 'number_3', 'number_4', 'number_5', 'number_6', 'super_number'])
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df.to_csv('lotto_numbers.csv', index=False)


if __name__ == '__main__':
    scraper()
