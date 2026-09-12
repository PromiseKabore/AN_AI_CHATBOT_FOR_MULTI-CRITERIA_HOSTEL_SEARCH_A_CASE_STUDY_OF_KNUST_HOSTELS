import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# --- Load and clean data ---
hostels = pd.read_csv("hostel_data.csv")
hostels.columns = [
    "S/N", "Hostel_Name", "Room_Type", "Price", "Distance",
    "Water", "Wifi", "Security", "TV_Room", "Study_Room",
]

def clean_text(x):
    return str(x).lower().replace(" ", "")

hostels["Room_Type"] = hostels["Room_Type"].apply(clean_text)

AMENITY_COLUMNS = {
    "water": "Water",
    "wifi": "Wifi",
    "security": "Security",
    "tv_room": "TV_Room",
    "study_room": "Study_Room",
}
AMENITY_LABELS = {
    "water": "water",
    "wifi": "wifi",
    "security": "security",
    "tv_room": "TV room",
    "study_room": "study room",
}

# --- Content-based profile (TF-IDF + cosine similarity) ---
# Same idea as the notebook's "soup": each hostel gets a short profile of
# room type + price tier + distance tier + amenities, and we rank candidates
# by how closely their profile matches the profile implied by the user's
# request, rather than only sorting by raw price/distance.

PRICE_TIERS = {
    "1in1": (7800, 8600),
    "2in1": (6800, 7800),
    "3in1": (5600, 7200),
    "4in1": (3000, 5800),
}
DEFAULT_PRICE_TIER = (4000, 7800)  # used when room type is unknown


def price_tier(price, room_type):
    low, mid = PRICE_TIERS.get(room_type, DEFAULT_PRICE_TIER)
    if price < low:
        return "low"
    elif price < mid:
        return "medium"
    return "high"


def distance_tier(mins):
    if mins <= 15:
        return "near"
    elif mins <= 30:
        return "mid"
    return "far"


def build_soup(row):
    tokens = [
        row["Room_Type"],
        price_tier(row["Price"], row["Room_Type"]),
        distance_tier(row["Distance"]),
    ]
    for key, col in AMENITY_COLUMNS.items():
        if row.get(col) == "Yes":
            tokens.append(key)
    return " ".join(tokens)


hostels["soup"] = hostels.apply(build_soup, axis=1)

_tfidf = TfidfVectorizer()
_tfidf_matrix = _tfidf.fit_transform(hostels["soup"])


def build_query_soup(room_type=None, max_price=None, max_distance=None, required_amenities=None):
    """Build a soup string representing what the user is looking for, so it
    can be compared against hostel profiles with cosine similarity."""
    tokens = []
    if room_type:
        tokens.append(room_type)
    if max_price is not None:
        tokens.append(price_tier(max_price, room_type))
    if max_distance is not None:
        tokens.append(distance_tier(max_distance))
    if required_amenities:
        tokens.extend(required_amenities)
    return " ".join(tokens)


def rank_by_similarity(candidates, query_soup):
    """Return candidates with a Similarity column, ranked by cosine
    similarity of each hostel's profile to the query profile. Falls back to
    leaving order untouched if the query soup is empty."""
    if candidates.empty or not query_soup.strip():
        candidates = candidates.copy()
        candidates["Similarity"] = 0.0
        return candidates

    query_vec = _tfidf.transform([query_soup])
    candidate_vecs = _tfidf_matrix[candidates.index]
    sims = cosine_similarity(query_vec, candidate_vecs)[0]

    candidates = candidates.copy()
    candidates["Similarity"] = sims
    return candidates


def hostel_to_text(row, show_room_type=True, show_price=True, show_distance=True, show_amenities=None):
    """Build a listing line for a hostel, only including the fields that
    are actually relevant to what the user asked about."""
    details = []
    if show_room_type:
        details.append(f"{row['Room_Type']}")
    if show_price:
        details.append(f"GHS {row['Price']}")
    if show_distance:
        details.append(f"{row['Distance']} min walk from campus")
    if show_amenities:
        present = [AMENITY_LABELS[a] for a in show_amenities if row.get(AMENITY_COLUMNS[a]) == "Yes"]
        if present:
            details.append(", ".join(present))

    if details:
        return f"{row['Hostel_Name']} ({', '.join(details)})"
    return f"{row['Hostel_Name']}"


def recommend_by_preferences(room_type=None, max_price=None, max_distance=None, top_n=5,
                              sort_order="asc", required_amenities=None):
    candidates = hostels.copy()

    if room_type is not None:
        room_type_clean = room_type.lower().replace(" ", "")
        candidates = candidates[candidates["Room_Type"] == room_type_clean]

    if max_price is not None:
        candidates = candidates[candidates["Price"] <= max_price]

    if max_distance is not None:
        candidates = candidates[candidates["Distance"] <= max_distance]

    if required_amenities:
        for a in required_amenities:
            col = AMENITY_COLUMNS.get(a)
            if col:
                candidates = candidates[candidates[col] == "Yes"]

    if candidates.empty:
        return None

    # Rank by cosine similarity to the profile implied by the request
    # (room type + price tier + distance tier + amenities), then use
    # price/distance as the tiebreaker.
    query_soup = build_query_soup(
        room_type=room_type,
        max_price=max_price,
        max_distance=max_distance,
        required_amenities=required_amenities,
    )
    candidates = rank_by_similarity(candidates, query_soup)

    ascending = (sort_order != "desc")
    candidates = candidates.sort_values(
        by=["Similarity", "Price", "Distance"],
        ascending=[False, ascending, ascending],
    )

    keep_cols = ["Hostel_Name", "Room_Type", "Price", "Distance"] + list(AMENITY_COLUMNS.values())
    result = candidates[keep_cols]
    if top_n is not None:
        result = result.head(top_n)
    return result


def filter_candidates(room_type=None, max_distance=None, required_amenities=None):
    """Used for stat queries (mode/median) — filters by room type / distance
    (and optionally amenities) only, since price itself is the thing being asked about."""
    candidates = hostels.copy()

    if room_type is not None:
        room_type_clean = room_type.lower().replace(" ", "")
        candidates = candidates[candidates["Room_Type"] == room_type_clean]

    if max_distance is not None:
        candidates = candidates[candidates["Distance"] <= max_distance]

    if required_amenities:
        for a in required_amenities:
            col = AMENITY_COLUMNS.get(a)
            if col:
                candidates = candidates[candidates[col] == "Yes"]

    return candidates


def price_stat(candidates, stat_type):
    """Compute mode or median price for a set of candidate hostels."""
    if candidates is None or candidates.empty:
        return None

    if stat_type == "mode":
        modes = candidates["Price"].mode()
        if modes.empty:
            return None
        return sorted(modes.tolist())

    if stat_type == "median":
        return candidates["Price"].median()

    return None


def extract_filters_with_ai(user_message):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """You are a hostel search assistant for a university in Ghana.
                Extract search filters and return ONLY a JSON object with these keys:
                - is_hostel_query: true or false. Set to true ONLY if the user is asking to find/search
                  for a hostel or accommodation based on room type, price/budget, or distance/proximity
                  to campus, OR asking a statistics question about hostel prices (e.g. most common price,
                  mid-range price). Set to false for anything else — e.g. requests about pharmacies,
                  restaurants, shops, transport, other campus services, general chit-chat, or questions
                  unrelated to hostels. If false, the other fields can be null.

                - stat_type: "mode" if the user is asking for the MOST COMMON price (e.g. "what's the
                  most common price", "what price comes up the most", "most typical price"). "median" if
                  the user mentions MID-RANGE, MIDDLE, AVERAGE, or "TYPICAL" price/hostels in ANY form
                  (e.g. "I want a mid range", "mid-range hostels", "middle price", "average price",
                  "what's a typical price") — always treat "mid range" as a request for the single median
                  price number, NEVER as a price ceiling for a listing. null if the user is asking for a
                  normal hostel listing/recommendation using cheap/expensive/budget language or specific
                  filters, not a mid-range or statistic request.

                - room_type: one of '1in1','2in1','3in1','4in1', or null
                - max_price: a number in GHS or null
                - max_distance: a number in WALKING MINUTES to campus or null
                - requested_count: how many results the user wants. Use the string "all" if they ask for
                  all/every hostel or all matching options. Use a number if they ask for a specific count
                  (e.g. "show me 10", "give me 3 options"). Use null if they didn't specify a count.
                - sort_order: "desc" if the user wants the most expensive / priciest / highest-priced
                  hostels first, "asc" if they want the cheapest / most affordable first. Phrases like
                  "most expensive", "priciest", "highest price" → "desc". Phrases like "cheap",
                  "cheapest", "affordable", "budget" → "asc". Default to "asc" if unclear. If the user
                  asks for "most expensive" or similar, leave max_price as null — use sort_order instead,
                  do not invent a price ceiling for expensive requests.
                - summary: a short, natural phrase describing what the user is searching for (e.g. "cheap hostels", "a single room close to campus", "affordable 2-in-1 rooms", "the most expensive hostels", "the most common hostel price")

                - mentions_room_type: true if the user explicitly referenced a room type (e.g. "1in1",
                  "single room", "2 in 1", "shared room"), false otherwise.
                - mentions_price: true if the user explicitly referenced price/cost/budget (e.g. "cheap",
                  "expensive", "under GHS X", "budget", "most common price", "mid-range price"), false
                  otherwise.
                - mentions_distance: true if the user explicitly referenced distance/proximity to campus
                  (e.g. "close", "near", "walking distance", "far", "X minutes away"), false otherwise.

                - required_amenities: a list containing any of "water", "wifi", "security", "tv_room",
                  "study_room" that the user explicitly asked for (e.g. "with wifi", "needs constant
                  water", "with security", "has a tv room", "with a study room/reading room"). Empty
                  list if none mentioned.
                - mentions_amenities: true if the user mentioned any amenity (water, wifi, security,
                  tv room, study room), false otherwise.

                - wants_list: ONLY relevant when stat_type is "mode" or "median". true if the user wants
                  actual hostel options/recommendations at that price point (e.g. "I want a mid-range
                  hostel", "show me mid-range hostels", "find me a hostel around the average price",
                  "recommend a mid-range hostel") — i.e. they used words like "hostel(s)", "room(s)",
                  "show me", "find me", "recommend", "give me options". false if the user is purely
                  asking for the number/statistic itself and nothing else (e.g. "what is the most common
                  price", "what's the median price", "what's the mid-range price", "tell me the average
                  price"). If stat_type is null, set wants_list to true.

                Distance guidance:
                - "close", "near", "walking distance" → max_distance: 15
                - "not too far", "fairly close"        → max_distance: 30
                - "far is okay", no mention of distance → max_distance: null

                Price guidance (set max_price based on BOTH budget word AND room type together,
                ONLY when the user is asking for cheap/budget options in a normal listing — never set
                max_price when the user asks for "most expensive" or similar, use sort_order for that
                instead, and never set max_price when stat_type is "mode" or "median". Do NOT use a
                "mid" price tier at all — any mid-range/average/typical language should set stat_type
                to "median" instead, as described above):
                - 1in1: cheap → 7800  | any/unmentioned → null
                - 2in1: cheap → 6800  | any/unmentioned → null
                - 3in1: cheap → 5600  | any/unmentioned → null
                - 4in1: cheap → 3000  | any/unmentioned → null
                - room type unknown: cheap → 4000   | any/unmentioned → null

                IMPORTANT: This system has hostel room/price/distance data AND these five in-hostel
                amenities: water, wifi, security, tv_room, study_room. Requests about these are valid
                hostel queries (is_hostel_query: true). But if the user mentions ANY location, amenity,
                or service that is NOT one of the hostel itself or the five amenities above (e.g. "close
                to a pharmacy", "near a supermarket", "next to a bank"), set is_hostel_query to false,
                since this system has no data on proximity to external services.

                Return ONLY the JSON, no extra text, no explanation."""
            },
            {"role": "user", "content": user_message}
        ]
    )

    text = response.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "is_hostel_query": False,
            "stat_type": None,
            "room_type": None,
            "max_price": None,
            "max_distance": None,
            "requested_count": None,
            "sort_order": "asc",
            "summary": "your request",
            "mentions_room_type": False,
            "mentions_price": False,
            "mentions_distance": False,
            "required_amenities": [],
            "mentions_amenities": False,
            "wants_list": True,
        }


def ai_chatbot(user_message):
    filters = extract_filters_with_ai(user_message)

    if not filters.get("is_hostel_query", False):
        return ("Sorry, I don't have information about that!\n\n"
                "I can help you find hostels based on room type, price, and "
                "walking distance from campus.")

    room_type       = filters.get("room_type")
    max_price       = filters.get("max_price")
    max_distance    = filters.get("max_distance")
    summary_text    = filters.get("summary", "your request")
    requested_count = filters.get("requested_count")
    sort_order      = filters.get("sort_order", "asc")
    stat_type       = filters.get("stat_type")

    show_room_type    = filters.get("mentions_room_type", False)
    show_price        = filters.get("mentions_price", False)
    show_distance     = filters.get("mentions_distance", False)
    mentions_amenities = filters.get("mentions_amenities", False)
    required_amenities = filters.get("required_amenities") or []
    show_amenities     = required_amenities if mentions_amenities else None

    # Fallback: if the AI didn't flag any field as mentioned, show everything
    # (keeps old behavior for vague queries like "recommend a hostel").
    if not (show_room_type or show_price or show_distance or mentions_amenities):
        show_room_type = show_price = show_distance = True

    if sort_order not in ("asc", "desc"):
        sort_order = "asc"

    # --- How many results were requested (used by both stat-listing and normal listing) ---
    if isinstance(requested_count, str):
        rc = requested_count.strip().lower()
        if rc == "all":
            top_n = None
        else:
            try:
                top_n = int(rc)
            except ValueError:
                top_n = 5
    elif isinstance(requested_count, (int, float)) and requested_count > 0:
        top_n = int(requested_count)
    else:
        top_n = 5

    # --- Statistic queries (mode / median price) ---
    if stat_type in ("mode", "median"):
        candidates = filter_candidates(room_type=room_type, max_distance=max_distance,
                                        required_amenities=required_amenities)
        stat_value = price_stat(candidates, stat_type)

        if stat_value is None:
            return "I couldn't find enough data to work that out."

        qualifier = ""
        if room_type:
            qualifier = f" for {room_type} rooms"
        elif max_distance:
            qualifier = f" within {max_distance} minutes of campus"

        wants_list = filters.get("wants_list", False)

        if not wants_list:
            if stat_type == "mode":
                if len(stat_value) == 1:
                    price_text = f"GHS {stat_value[0]}"
                else:
                    price_text = "GHS " + " and GHS ".join(str(p) for p in stat_value)
                return f"The most common hostel price{qualifier} is {price_text}."
            else:  # median
                price_text = f"GHS {stat_value:.2f}" if stat_value % 1 else f"GHS {int(stat_value)}"
                return f"The mid-range (median) hostel price{qualifier} is {price_text}."

        # wants_list=True: return actual hostels centered on the stat value
        matched = candidates.copy()
        if stat_type == "mode":
            matched = matched[matched["Price"].isin(stat_value)]
            matched = matched.sort_values(by=["Price", "Distance"])
        else:  # median
            matched["_diff"] = (matched["Price"] - stat_value).abs()
            matched = matched.sort_values(by=["_diff", "Distance"])

        if top_n is not None:
            matched = matched.head(top_n)

        if matched.empty:
            return f"I couldn't find any hostels matching that price range{qualifier}."

        listings = "\n\n".join(
            matched.apply(
                lambda row: hostel_to_text(
                    row,
                    show_room_type=show_room_type,
                    show_price=True,
                    show_distance=show_distance,
                    show_amenities=show_amenities,
                ),
                axis=1
            )
        )
        return f"Understood! Searching for {summary_text}\n\n{listings}"

    # --- Normal listing queries ---
    intro = f"Understood! Searching for {summary_text}\n\n" if summary_text else "Sorry I have no information about that!\n\n"

    result = recommend_by_preferences(
        room_type=room_type,
        max_price=max_price,
        max_distance=max_distance,
        top_n=top_n,
        sort_order=sort_order,
        required_amenities=required_amenities,
    )

    if result is not None and len(result) > 0:
        listings = "\n\n".join(
            result.apply(
                lambda row: hostel_to_text(
                    row,
                    show_room_type=show_room_type,
                    show_price=show_price,
                    show_distance=show_distance,
                    show_amenities=show_amenities,
                ),
                axis=1
            )
        )
    else:
        listings = "No hostels found matching your criteria."

    return intro + listings