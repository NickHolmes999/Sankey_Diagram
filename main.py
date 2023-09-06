import csv
import os
import re
import time
import tkinter as tk
import webbrowser
from datetime import datetime
from itertools import chain
from tkinter import StringVar, OptionMenu, Label, Button, Text

import folium
import geojson
import geopandas as gpd
import h3
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import openrouteservice
import openrouteservice.directions
import openrouteservice.exceptions
import pandas as pd
import plotly.graph_objects as go
import webview
from folium.plugins import HeatMap
from folium.plugins import MarkerCluster
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from polyline import polyline
from shapely.geometry import Polygon

color_list = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta']
color_dict = {}
def concat_events(series):
    return ''.join(series)
# Read the Excel file again into a new DataFrame
df_original = pd.read_excel('venv/PythonExcelExample.xlsx')
df_original['Purchased Item'] = df_original['Event'].shift(+2).str.strip()

# Filter out the rows where the 'Event' column is 'Summary'
df_purchases = df_original[df_original['Event'] == 'Summary']
top_selling_items = df_purchases['Purchased Item'].value_counts().nlargest(10)

# Filter out the events that are not 'Summary'
df_summary_original = df_original[df_original['Event'] == 'Summary']

df = pd.read_excel('venv/PythonExcelExample.xlsx')
# Convert 'Time' column to datetime.timedelta objects
df['Time'] = df['Time'].apply(lambda x: datetime.combine(datetime.min, x) - datetime.min)
# Create 'EventPath' and 'LagTime' columns in 'df'
df['EventPath'] = df.groupby('ID')['Event'].transform(concat_events)
df['LagTime'] = df.groupby('ID')['Time'].diff().fillna(pd.Timedelta(seconds=0))
# Shift the 'LagTime' values by one position
df['LagTime'] = df.groupby('ID')['LagTime'].shift(-1).fillna(pd.Timedelta(seconds=0))
# Count the events for each row
df['EventCount'] = df.groupby('ID').cumcount() + 1
# Prepare the 'output_data' list
output_data = []
for id, group in df.groupby('ID'):
    time_lag_pairs = group[['LagTime', 'Event']].values
    output_data.append([id, group.iloc[0]['Name']] + list(chain.from_iterable(time_lag_pairs)))
# Sort the output data based on the 'EventCount' and 'EventPath' columns
output_data_sorted = sorted(output_data, key=lambda x: (len(x[2:]), ''.join(x[3::2])))
# Write the resulting output data to a CSV file
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in output_data_sorted:
        writer.writerow(row)
##########################################################################################################
##########################################################################################################
def retrieve_user_paths(df):
    grouped_df = df.groupby('ID')
    user_paths = [{'id': _, 'name': group['Name'].iat[0], 'path': list(group['Event'])} for _, group in grouped_df]
    return user_paths
def generate_nodes_links(user_paths):
    nodes = []  # Keep this as a list
    nodes_set = set()  # Use a set for efficient membership checking
    links = {'source': [], 'target': [], 'value': [], 'label': [], 'customdata': [], 'color':[]}
    colors = list(mcolors.CSS4_COLORS.keys())  # Get a list of available color names
    color_dict = {}  # Create an empty dictionary to store unique colors for each path
    for user in user_paths:
        path_str = '-'.join(user['path'])
        if path_str not in color_dict.keys():  # If this path doesn't have a unique color assigned
            color_dict[path_str] = colors.pop()  # Assign and remove one from the list
        for i in range(len(user['path']) - 1):
            if user['path'][i] not in nodes_set:
                nodes.append(user['path'][i])
                nodes_set.add(user['path'][i])
            if user['path'][i + 1] not in nodes_set:
                nodes.append(user['path'][i + 1])
                nodes_set.add(user['path'][i + 1])
            links['source'].append(nodes.index(user['path'][i]))
            links['target'].append(nodes.index(user['path'][i + 1]))
            links['value'].append(1)
            links['label'].append(path_str)
            links['customdata'].append(f"ID: {user['id']}<br>Name: {user['name']}")
            links['color'].append(color_dict[path_str])  # Add the color to the link
    return nodes, links
# Create a Sankey Diagram
def create_sankey(df):
    user_paths = retrieve_user_paths(df)
    nodes, links = generate_nodes_links(user_paths)
    data = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="blue",
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=links['color'], # Set link color
            hovertemplate='Link from node %{source.label}<br />' +
                          'to node %{target.label} has value %{value}<br />' +
                          '%{customdata}<extra></extra>',
            customdata=links['customdata'],
        )
    )
    # Step 1: Compute the unique path occurrences
    paths = df.groupby('EventPath')['ID'].nunique().sort_values(ascending=False)
    # Step 2: Extract the top 5 unique event paths
    top5_paths = paths.head(5)
    # Prepare top 5 paths for annotation
    annotations = []
    for i, (path, count) in enumerate(top5_paths.items()):
        # Insert dashes between events in 'path'. Assumes events always start with a capital letter.
        dashed_path = re.sub(r"(?<=\w)([A-Z])", r"-\1", path)
        annotations.append(
            dict(
                text=f"Path: {dashed_path}, Count: {count}",
                xref="paper", yref="paper",
                x=0.95, y=(1 - 0.05 * i),  # Positions to stack annotations nicely
                showarrow=False
            )
        )
    layout = go.Layout(title_text="Sankey Diagram", font_size=10, annotations=annotations)
    fig = go.Figure(data=data, layout=layout)
    fig.show()
##########################################################################################################
# Create a new DataFrame from the rows containing 'summary'
purchase_data = df[df['Event'] == 'Summary'].copy()
geolocator = Nominatim(user_agent="Nick-Holmes_Practice", timeout=15)
client = openrouteservice.Client(key='5b3ce3597851110001cf6248d3b6653cdffb4a26977343a5ebebc5fb')
class Application:
    def __init__(self, master, geolocator, purchase_data):
        self.master = master
        self.master.title("Excel Sheet Analysis")
        self.geolocator = geolocator
        self.purchase_data = purchase_data
        self.search_path = []
        self.search_label_string = StringVar()
        self.search_label = Label(master, textvariable=self.search_label_string)
        self.search_label.pack()
        self.options = sorted(set(df['Event']))
        self.selected_option = StringVar(master)
        self.selected_option.set(self.options[0])  # default value
        self.dropdown_menu = OptionMenu(master, self.selected_option, *self.options)
        self.dropdown_menu.pack()
        self.add_button = Button(master, text="Add event to search path", command=self.add_to_search_path)
        self.add_button.pack()
        self.remove_last_button = Button(master, text="Remove last event", command=self.remove_last)
        self.remove_last_button.pack()
        self.clear_button = Button(master, text="Clear search path", command=self.clear_search_path)
        self.clear_button.pack()
        self.end_in_button = Button(master, text="End In", command=self.toggle_end_in)
        self.end_in_button.configure(relief="raised")
        self.end_in_button.pack()
        self.search_button = Button(master, text="Search", command=self.search)
        self.search_button.pack()
        self.results_text = Text(master)
        self.results_text.pack()
        self.results_text.tag_configure("result", background="white", foreground="green")
        self.results_text.tag_configure("separator", background="white", foreground="blue")
        self.sankey_button = Button(self.master, text="Show Sankey Diagram", command=lambda: create_sankey(df))
        self.sankey_button.pack()
        self.heatmap_button = Button(master, text="Show Heatmap", command=self.create_map)
        self.heatmap_button.pack()
        self.drive_time_button = Button(master, text="Open Drive Time", command=self.calculate_drive_time)
        self.drive_time_button.pack()
        # Control variable for the End In mode
        self.end_in_mode = False
        # Geolocate the purchase data
        result = purchase_data.apply(
            lambda row: self.geolocate(row['Country'], row['Region'], row['City'], row['Mailing Address']), axis=1)
        result_list = list(zip(*result))
        if result_list:
            purchase_data['Latitude'], purchase_data['Longitude'] = result_list
        else:
            purchase_data['Latitude'], purchase_data['Longitude'] = np.nan, np.nan
        # Count the frequency of each location
        self.location_counts = purchase_data.groupby(['Latitude', 'Longitude']).size().reset_index(name='Counts')


        # Compute the statistics
        self.most_common_city = df_summary_original['City'].mode()[0]
        self.most_common_region = df_summary_original['Region'].mode()[0]
        self.most_common_country = df_summary_original['Country'].mode()[0]
        self.most_common_race = df_summary_original['Race'].mode()[0]

        # Extract the hour from the 'Time' column
        self.most_common_hour = df_summary_original['Time'].apply(lambda x: x.hour)
        self.most_common_hour = self.hour_to_string(self.most_common_hour.mode()[0])
        self.stats_button = tk.Button(self.master, text="Show stats", command=self.show_stats)
        self.stats_button.pack()
    def add_to_search_path(self):
        self.search_path.append(self.selected_option.get())
        self.search_label_string.set(' > '.join(self.search_path))
    def remove_last(self):
        if self.search_path:
            self.search_path.pop()
            self.search_label_string.set(' > '.join(self.search_path))
    def clear_search_path(self):
        self.search_path.clear()
        self.search_label_string.set('')
    def toggle_end_in(self):
        self.end_in_mode = not self.end_in_mode
        print(f"Toggle end_in_mode: {self.end_in_mode}")  # Debug print
        if self.end_in_mode:
            self.end_in_button.config(bg="green", fg="white")
        else:  # when the button is pressed again and the end in mode is off.
            self.end_in_button.config(bg="SystemButtonFace", fg="black")
    def search(self):
        print(f"Searching, end_in_mode: {self.end_in_mode}, search_path: {self.search_path}")  # Debug Print
        self.results_text.delete(1.0, 'end')
        all_results = sorted(
            [(len(set(self.search_path).difference(set(row[3::2]))), row)
             for row in output_data
             if any(event in self.search_path for event in row[3::2]) and
             (not self.end_in_mode or
              (len(self.search_path) <= len(row[3::2]) and
               tuple(row[3::2][-len(self.search_path):]) == tuple(self.search_path)))
             ],
            key=lambda x: x[0]
        )
        for num_diff, result in all_results:
            self.results_text.insert('end', '-' * 50 + '\n', 'separator')
            self.results_text.insert('end', str(result) + '\n', 'result')
        self.results_text.insert('end', '-' * 50 + '\n', 'separator')
    def geolocate(self, country, region, city, address):
        # Check if any values are nan
        if pd.isna(country) or pd.isna(region) or pd.isna(city) or pd.isna(address):
            return np.nan, np.nan  # return NaN for rows with missing values
        # Combine the address, city, region, and country
        location_str = f"{address}, {city}, {region}, {country}"
        print(f"Geolocating: {location_str}")  # Debug print
        try:
            time.sleep(2)  # Add a 1 second delay between each request
            location = self.geolocator.geocode(location_str)
            if location != None:
                print(f"Found location: {location.latitude}, {location.longitude}")  # Debug print
                return location.latitude, location.longitude
            else:
                print(f"No location found for {location_str}")  # Debug print
                return np.nan, np.nan  # return NaN if location can't be found
        except Exception as e:
            print(f"Error geolocating {location_str}: {e}")
            return np.nan, np.nan  # return NaN if there's an error
    def create_map(self):
        self.location_counts['Latitude'] = pd.to_numeric(self.location_counts['Latitude'], errors='coerce')
        self.location_counts['Longitude'] = pd.to_numeric(self.location_counts['Longitude'], errors='coerce')
        # Calculate the mean latitude and longitude
        mean_latitude = self.location_counts['Latitude'].mean(skipna=True)
        mean_longitude = self.location_counts['Longitude'].mean(skipna=True)
        # Check if the mean latitude and longitude are NaN
        if pd.isna(mean_latitude) or pd.isna(mean_longitude):
            # If they are NaN, set a default location
            location = [0, 0]  # You can set your own default location here
        else:
            # If they are not NaN, use the mean latitude and longitude
            location = [mean_latitude, mean_longitude]
        # Create a folium map centered at the location
        m = folium.Map(location=location, zoom_start=5, zoom_control=True, tiles='OpenStreetMap')
        tile = folium.TileLayer(tiles='OpenStreetMap', opacity=0.85)
        tile.add_to(m)
        # Add a heatmap to the map
        HeatMap(data=self.location_counts[['Latitude', 'Longitude', 'Counts']].dropna(), radius=15, max_zoom=10).add_to(m)
        marker_cluster = MarkerCluster().add_to(m)
        # Add markers to the map
        for index, row in self.location_counts.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color="red",
                fill=True,
                fill_color="red",
                tooltip=f"Lat: {row['Latitude']}, Lon: {row['Longitude']}"
            ).add_to(marker_cluster)
        # Save the map as an HTML string
        map_html = m._repr_html_()
        # Create a new webview window with the map HTML
        webview.create_window('Map', html=map_html)
        webview.start()

    def calculate_drive_time(self):
        warehouse_coords = [39.048231, -77.113589]

        m = folium.Map(location=warehouse_coords, zoom_start=5, zoom_control=True, tiles='OpenStreetMap')

        # Create a MarkerCluster object
        marker_cluster = MarkerCluster()

        # Add the warehouse marker with custom color and size
        folium.Marker(warehouse_coords, popup="Warehouse", icon=folium.Icon(color='red', icon='home', prefix='fa'),
                      icon_color='white', icon_size=(40, 40)).add_to(marker_cluster)

        client = openrouteservice.Client(key='5b3ce3597851110001cf6248d3b6653cdffb4a26977343a5ebebc5fb')

        # Create a GeoDataFrame from your DataFrame
        gdf = gpd.GeoDataFrame(
            self.purchase_data, geometry=gpd.points_from_xy(self.purchase_data.Longitude, self.purchase_data.Latitude)
        )

        # Convert latitude and longitude to H3 hexagons
        self.purchase_data['h3'] = self.purchase_data.apply(
            lambda row: h3.geo_to_h3(row['Latitude'], row['Longitude'], resolution=7), axis=1)

        # Group by the 'h3' column and count the number of points in each bin
        hexbins = self.purchase_data.groupby('h3').size().reset_index(name='count')

        # Convert the 'h3' column back to latitude and longitude
        hexbins['geometry'] = hexbins['h3'].apply(lambda x: Polygon(h3.h3_to_geo_boundary(x, geo_json=True)))

        # Convert the DataFrame to a GeoDataFrame
        hexbins = gpd.GeoDataFrame(hexbins, geometry='geometry')

        # Normalize the count column to get values between 0 and 1
        hexbins['count_normalized'] = (hexbins['count'] - hexbins['count'].min()) / (
                hexbins['count'].max() - hexbins['count'].min())

        # Import the necessary libraries
        import matplotlib.colors as colors

        # Define the colormap to use
        colormap = matplotlib.cm.get_cmap('coolwarm') # without `len(hexbins)`



        # Calculate the maximum count of points
        max_count = hexbins['count'].max()
        min_count = hexbins['count'].min()

        # Define the number of bins
        num_bins = 10

        # Create the bins
        bins = np.linspace(hexbins['count'].min(), hexbins['count'].max(), num_bins)

        # Categorize the counts into the bins
        hexbins['count_bin'] = np.digitize(hexbins['count'], bins)

        for index, row in self.purchase_data.iterrows():
            if row['Event'] == 'Summary':
                buyer_coords = [row['Latitude'], row['Longitude']]

                distance = geodesic(warehouse_coords, buyer_coords).meters
                print(f"Distance: {distance} meters")

                try:
                    route = client.directions([warehouse_coords[::-1], buyer_coords[::-1]], radiuses=[350, 350])
                except openrouteservice.exceptions.ApiError as e:
                    print(f"Error calculating route for coordinates {warehouse_coords} and {buyer_coords}: {e}")
                    continue

                drive_time = route['routes'][0]['summary']['duration'] / 60

                # Get the 'id' and 'name' from the row
                buyer_id = row['ID']
                buyer_name = row['Name']

                # Create a string with the drive time, id, and name
                popup_text = f"Drive Time: {drive_time:.2f} minutes<br>ID: {buyer_id}<br>Name: {buyer_name}"

                folium.Marker(buyer_coords, popup=popup_text).add_to(marker_cluster)
                print(route)

                # Decode the polyline to get the coordinates
                geometry = route['routes'][0]['geometry']
                coords = polyline.decode(geometry)
                # Reverse each coordinate and build the locations list
                locations = [list(reversed(coord)) for coord in coords]

                folium.PolyLine(
                    locations=locations,
                    color='blue').add_to(m)

        for idx, row in hexbins.iterrows():
            # Create a GeoJSON polygon
            polygon = geojson.Polygon([list(row.geometry.exterior.coords)])

            # Count the number of points within the polygon
            points_in_polygon = row['count']
            print(f"Polygon {idx + 1} contains {points_in_polygon} points.")

            # Calculate the color based on the count_normalized
            color = colors.rgb2hex(colormap(row['count_bin']/num_bins))


            # Add the polygon to the map
            folium.GeoJson(
                polygon,
                name='geojson',
                style_function=lambda x: {'fillColor': color},
            ).add_to(m)

        marker_cluster.add_to(m)

        m.save('drive_time_map.html')
        webbrowser.open_new_tab('file://' + os.path.abspath('drive_time_map.html'))

    def show_stats(self):
        stats_window = tk.Toplevel(self.master)
        stats_label = tk.Label(stats_window, text=f'Most common city: {self.most_common_city}\n'
                                                  f'Most common region: {self.most_common_region}\n'
                                                  f'Most common country: {self.most_common_country}\n'
                                                  f'Most common hour: {self.most_common_hour}\n'
                                                  f'Most common race: {self.most_common_race}\n'
                                                  f'Most popular item: {top_selling_items.index[0]}\n'
                                                  f'Second most popular item: {top_selling_items.index[1]}\n')
        stats_label.pack()

    def hour_to_string(self, hour):
        if 0 <= hour < 6:
            return 'Night'
        elif 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        else:
            return 'Evening'

root = tk.Tk()
app = Application(root, geolocator, purchase_data)
root.mainloop()