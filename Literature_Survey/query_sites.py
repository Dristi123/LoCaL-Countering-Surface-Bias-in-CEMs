from serpapi import GoogleSearch
import feedparser
import json
import csv

API_KEY = "" # update yours here (SerpAPI API Key)

metrics = ["CodeBLEU", "CrystalBLEU", "CodeBERTScore", "CodeScore"]
years = ["2021", "2022", "2023", "2024", "2025"]

seen_titles = set()
results = []

def is_english(text):
    try:
        text.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False

print("======Searching Google Scholar=======")
for metric in metrics:
    for year in years:
        params = {
            "engine": "google_scholar",
            "q": metric,
            "as_ylo": year,
            "as_yhi": year,
            "num": "20",
            "api_key": API_KEY
        }

        search = GoogleSearch(params)
        data = search.get_dict()

        for item in data.get("organic_results", []):
            title = item.get("title", "").strip()
            key = (title.lower(), metric.lower())
            if title and is_english(title) and key not in seen_titles:
                seen_titles.add(key)
                authors = [a.get("name") for a in item.get("publication_info", {}).get("authors", [])]
                results.append({
                    "source": "Google Scholar",
                    "metric": metric,
                    "title": title,
                    "link": item.get("link", ""),
                    "authors": ", ".join(authors)
                    
                })

print("============Searching arXiv============")
base_url = "http://export.arxiv.org/api/query?search_query=all:{}&start={}&max_results={}"
for metric in metrics:
    for start in range(0, 100, 25):
        query = base_url.format(metric, start, 25)
        feed = feedparser.parse(query)
        for entry in feed.entries:
            title = entry.title.strip()
            key = (title.lower(), metric.lower())
            if title and is_english(title) and key not in seen_titles:
                seen_titles.add(key)
                authors = [author.name for author in entry.authors]
                results.append({
                    "source": "arXiv",
                    "metric": metric,
                    "title": title,
                    "link": entry.link,
                    "authors": ", ".join(authors),
                 
                })




with open("metric_mentions.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["title", "link", "authors", "metric"])
    writer.writeheader()
    for row in results:
        writer.writerow({
            "title": row["title"],
            "link": row["link"],
            "authors": row["authors"],
            "metric": row["metric"],
        })
print(f"Saved {len(results)} entries to metric_raw_mentions.csv")


