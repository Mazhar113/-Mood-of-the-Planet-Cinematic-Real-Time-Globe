import tweepy
from textblob import TextBlob
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.graph_objects as go
from threading import Thread
import time
import numpy as np
from sklearn.cluster import DBSCAN
import math

# -----------------------------
# 1. Twitter API Setup
# -----------------------------
bearer_token = "YOUR_BEARER_TOKEN"
client = tweepy.Client(bearer_token=bearer_token)

# -----------------------------
# 2. Sentiment Analysis
# -----------------------------
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "Happy"
    elif polarity < -0.2:
        return "Sad"
    else:
        return "Neutral"

emoji_map = {"Happy": "😊", "Sad": "😢", "Neutral": "😐"}
color_map = {"Happy": "yellow", "Sad": "blue", "Neutral": "red"}

# -----------------------------
# 3. Data Storage
# -----------------------------
tweets_df = pd.DataFrame(columns=["lat", "lon", "sentiment", "text", "timestamp"])

# -----------------------------
# 4. Background Tweet Fetching
# -----------------------------
def fetch_tweets():
    global tweets_df
    while True:
        try:
            query = " -is:retweet lang:en has:geo"
            tweets = client.search_recent_tweets(
                query=query,
                tweet_fields=["geo","text","created_at"],
                expansions=["geo.place_id"],
                max_results=10
            )
            if not tweets.data:
                time.sleep(5)
                continue
            places = {place["id"]: place for place in tweets.includes["places"]} if tweets.includes and "places" in tweets.includes else {}
            for tweet in tweets.data:
                if tweet.geo and tweet.geo.get("place_id") in places:
                    place = places[tweet.geo["place_id"]]
                    bbox = place["geo"]["bbox"]
                    lat = (bbox[1] + bbox[3]) / 2
                    lon = (bbox[0] + bbox[2]) / 2
                    sentiment = analyze_sentiment(tweet.text)
                    timestamp = time.time()
                    tweets_df.loc[len(tweets_df)] = [lat, lon, sentiment, tweet.text, timestamp]
        except Exception as e:
            print("Error fetching tweets:", e)
        time.sleep(5)

thread = Thread(target=fetch_tweets)
thread.daemon = True
thread.start()

# -----------------------------
# 5. Dash App Setup
# -----------------------------
app = dash.Dash(__name__)
app.title = "🌍 Ultimate Cinematic Mood of the Planet"

app.layout = html.Div([
    html.H1("🌍 Ultimate Cinematic Mood of the Planet", style={'textAlign':'center'}),
    dcc.Graph(id='globe-graph', style={'height':'100vh', 'width':'100vw'}),
    dcc.Interval(id='interval', interval=500)  # update every 0.5s for smooth animation
])

# -----------------------------
# 6. Comet Initialization
# -----------------------------
num_comets = 10
comets = {
    "lat": np.random.uniform(-90,90,num_comets),
    "lon": np.random.uniform(-180,180,num_comets),
    "speed": np.random.uniform(0.5,2.0,num_comets)
}

# -----------------------------
# 7. Callback for Full-Screen Cinematic Globe with Comets
# -----------------------------
@app.callback(
    Output('globe-graph', 'figure'),
    Input('interval', 'n_intervals')
)
def update_globe(n):
    global tweets_df, comets
    current_time = time.time()
    tweets_df = tweets_df[tweets_df['timestamp'] > current_time - 60]

    if tweets_df.empty:
        fig = go.Figure()
        fig.update_geos(projection_type='orthographic')
        fig.update_layout(paper_bgcolor='black', plot_bgcolor='black', font_color='white')
        return fig

    coords = tweets_df[['lat','lon']].to_numpy()
    if len(coords) > 1:
        clustering = DBSCAN(eps=3, min_samples=2).fit(coords)
        tweets_df['cluster'] = clustering.labels_
    else:
        tweets_df['cluster'] = 0

    cluster_markers = []
    heat_lats, heat_lons, heat_weights = [], [], []

    heartbeat_factor = 1 + 0.3 * math.sin(time.time()*3)

    for cluster_id in tweets_df['cluster'].unique():
        cluster_data = tweets_df[tweets_df['cluster']==cluster_id]
        lat = cluster_data['lat'].mean()
        lon = cluster_data['lon'].mean()
        count = len(cluster_data)
        sentiment = cluster_data['sentiment'].mode()[0]
        age = (current_time - cluster_data['timestamp']).mean()
        base_size = 20 + count*5 + 20*(1 - age/60)
        size = base_size * heartbeat_factor
        opacity = 0.3 + 0.7*(1 - age/60)

        cluster_markers.append(
            go.Scattergeo(
                lon=[lon],
                lat=[lat],
                text=f"{emoji_map[sentiment]} ({count} tweets)",
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=color_map[sentiment],
                    opacity=opacity,
                    line=dict(width=2, color='white')
                ),
                textposition='top center',
                hoverinfo='text'
            )
        )

        for _ in range(count*3):
            heat_lats.append(lat + np.random.uniform(-0.5,0.5))
            heat_lons.append(lon + np.random.uniform(-0.5,0.5))
            heat_weights.append(max(0.1, opacity))

    heat = go.Densitymapbox(
        lat=heat_lats,
        lon=heat_lons,
        z=heat_weights,
        radius=30,
        colorscale='Jet',
        opacity=0.5,
        showscale=False
    )

    fig = go.Figure(data=cluster_markers + [heat])

    # Smooth globe rotation
    fig.update_geos(
        projection_type='orthographic',
        showcoastlines=True, coastlinecolor='white',
        showland=True, landcolor='rgb(20,20,20)',
        showocean=True, oceancolor='rgb(0,0,10)',
        projection_rotation=dict(lon=(time.time()*5)%360)
    )

    fig.update_layout(
        title="🌍 Ultimate Cinematic Mood of the Planet - Live Pulse, Comets, Smooth Transitions",
        paper_bgcolor='black',
        plot_bgcolor='black',
        font_color='white',
        geo=dict(bgcolor='black'),
        margin=dict(l=0,r=0,t=40,b=0)
    )

    # Add stars
    star_lats = np.random.uniform(-90, 90, 500)
    star_lons = np.random.uniform(-180, 180, 500)
    fig.add_trace(go.Scattergeo(
        lat=star_lats,
        lon=star_lons,
        mode='markers',
        marker=dict(size=1, color='white'),
        showlegend=False,
        hoverinfo='none'
    ))

    # Update comet positions
    comets["lon"] = (comets["lon"] + comets["speed"]) % 360 - 180
    fig.add_trace(go.Scattergeo(
        lat=comets["lat"],
        lon=comets["lon"],
        mode='markers+lines',
        marker=dict(size=3, color='white'),
        line=dict(color='white', width=2),
        showlegend=False,
        hoverinfo='none'
    ))

    return fig

# -----------------------------
# 8. Run App
# -----------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
