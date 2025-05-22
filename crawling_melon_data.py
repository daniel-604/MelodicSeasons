from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
import time
import pandas as pd
import re
from bs4 import BeautifulSoup

options = webdriver.ChromeOptions()
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

service = ChromeService(executable_path=ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

def getDataframe(year):
    print(year)
    url = f'https://www.melon.com/chart/age/index.htm?chartType=YE&chartGenre=KPOP&chartDate={year}'
    driver.get(url)
    time.sleep(0.5)

    driver.execute_script('window.scrollTo(0,800)')
    
    html_source = driver.page_source
    soup = BeautifulSoup(html_source, 'html.parser')
    
    # get song id
    song_info = soup.find_all('div', {'class': 'ellipsis rank01'})
    time.sleep(0.5)
    songid_list = [
        re.sub('[^0-9]', '', a_tag['href'].split(',')[1])
        for sid in song_info
        if (a_tag := sid.find('a')) is not None and 'href' in a_tag.attrs
    ]
    ret = []
    
    for i, song_id in enumerate(songid_list):
        
        song_url = f'https://www.melon.com/song/detail.htm?songId={song_id}'
        driver.get(song_url)
        time.sleep(0.5)
        lyric = ''
        date = ''
        month = ''
        
        try:
            driver.find_element(By.CSS_SELECTOR, '.button_more.arrow_d').click()
            time.sleep(0.5)
            song_soup = BeautifulSoup(driver.page_source, 'html.parser')
            lyric = song_soup.select_one('.lyric')
            
            # lyric
            if lyric:
                clean_lyric = re.sub('<.*?>', '', str(lyric).replace('<br/>', ' EOS '))
                clean_lyric = re.sub(r'\s+', ' ', clean_lyric).strip()
                clean_lyric = re.sub('EOS', '<EOS>', clean_lyric)
                lyric = clean_lyric
            else:
                print(f"No lyrics found for {song_id}")
            
            # date
            info_items = song_soup.select('div.meta dl.list dt')
            time.sleep(0.5)
            flag = False
            for dt in info_items:
                time.sleep(0.5)
                if dt.text.strip() == '발매일':
                    release_dd = dt.find_next_sibling('dd')
                    if release_dd:
                        date = release_dd.text.strip()
                        month = date[5:7]
                        flag = True
                    break
            if not flag:
                print(f'release date not found for {song_id}')
            
        except:
            print(f'Error occurred for {song_id}')
    
        if date != '' and lyric != '' and month != '':
            ret.append({'song_id': song_id, 'year': year, 'date': date, 'month': month, 'lyric': lyric})
            
    return ret

for year in range(1964, 2025):
    df = pd.read_csv('data/song_data.csv')
    new_df = pd.DataFrame(getDataframe(year))
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(f'data/song_data.csv', index=False, encoding='utf-8-sig')
driver.close()