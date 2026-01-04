import re

def parse_query(q: str):
    filters = {}
    if not q:
        return filters

    # Extract number of images
    num = re.search(r"(\d+)", q)
    if num:
        filters["limit"] = int(num.group(1))

    # Detect category
    if "stage" in q.lower():
        filters["category"] = "stage"

    # Extract date
    date = re.search(r"(\d{1,2}\s\w+\s20\d{2})", q)
    if date:
        filters["event_date"] = date.group(1)

    # Extract event name (fallback)
    events = ["devfest", "hackathon", "marathon"]
    for e in events:
        if e in q.lower():
            filters["event_name"] = e.title()

    return filters
