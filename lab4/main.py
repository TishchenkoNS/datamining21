import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector
import requests
from tkinter import Tk, simpledialog

# https://translate.google.com/?hl=ru/

plt.figure(figsize=(12, 8))

Tk().withdraw()
user_input = simpledialog.askstring("Input", "Put the link as text")
D = 0.5
LINKS = []

def calculate_ranks(tmp, pages_ranks):
    for i in range(1000):
        if i > 0:
            pages_ranks = tmp
        for y in range(len(pages_ranks)):
            sum_accumulator = 0
            for page in pages_list:
                if (page, pages_list[y]) in unique_links:
                    sum_accumulator += pages_ranks[pages_list.index(page)] / pages_dictionary.get(page)
            tmp[y] = (1 - D) + D * sum_accumulator

    return pages_ranks

def build_graph(unique_links):
    graph = nx.DiGraph()
    graph.add_edges_from(unique_links)
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=200)
    nx.draw_networkx_edges(graph, pos, edgelist=graph.edges)
    nx.draw_networkx_labels(graph, pos)
    plt.show()


def get_encoding(resp):
    http_encoding = resp.encoding if 'charset' in resp.headers.get('content-type', '').lower() else None
    html_encoding = EncodingDetector.find_declared_encoding(resp.content, is_html=True)
    return html_encoding or http_encoding


def html_recursive_parser(url):
    for link in LINKS:
        if url == link[0]:
            return
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, 'html.parser', from_encoding=get_encoding(resp))
    links_list = []

    for lnk in soup.find_all('a', href=True):
        full_link = user_input[:-1] + lnk['href']
        if "http" not in lnk['href']:
            links_list.append(full_link)

    for lnk in links_list:
        LINKS.append((url, lnk))
        html_recursive_parser(lnk)


html_recursive_parser(user_input)
unique_links = list(set(LINKS))
build_graph(unique_links)

pages_dictionary = {}
for link in unique_links:
    pages_dictionary[link[0]] = pages_dictionary.get(link[0], 0) + 1

pages_list = list(pages_dictionary)
pages_ranks = np.full(len(pages_list), 1)
tmp = np.zeros(len(pages_ranks))
pages_ranks = calculate_ranks(tmp, pages_ranks)

top_ten_pages = []
for i in range(len(pages_ranks)):
    top_ten_pages.append([pages_list[i], pages_ranks[i]])

top_ten_pages = sorted(top_ten_pages, reverse=True, key=lambda x: x[1])[:10]

for page, rank in top_ten_pages:
    print(f'Page: {page}; Rank: {rank}')
