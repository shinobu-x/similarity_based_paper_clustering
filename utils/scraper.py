import os
import urllib.request
import urllib
import socket
from bs4 import BeautifulSoup
def scraper(base,target):
    url = base + '/' + target
    req = urllib.request.Request(url)
    html = urllib.request.urlopen(req)
    soup = BeautifulSoup(html, 'html.parser')
    paper_list = soup.find_all('a', text = 'pdf')
    urls = []
    for paper in paper_list:
        url = base + paper['href']
        urls.append(url)
    file='./data/paper_list.csv'
    with open(file,mode='w') as f:
        f.write('title\n')
        for url in urls:
            title=url.split('/')[-1].split('_')[1:-3]
            paper=''
            for word in title:
                paper+=word+' '
            paper=paper[0:-1]
            f.write(paper+'\n')

base='http://openaccess.thecvf.com'
target='CVPR2021?day=2021-06-21'
scraper(base,target)
