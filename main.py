import pandas as pd
import csv
from datetime import datetime
import tkinter as tk
from tkinter import *
from tkinter import StringVar, OptionMenu, Label, Button, Text
import matplotlib.colors as mcolors
import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget, QComboBox, QLabel, QTextEdit
import plotly.graph_objects as go
from plotly.offline import plot
import re
from folium.plugins import HeatMap
import folium
from io import BytesIO
import base64
import numpy as np
from geopy.geocoders import Nominatim




color_list = ['blue', 'green', 'red', 'yellow', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta']
color_dict = {}
def concat_events(series):
    return ''.join(series)

# Read the Excel file into a Pandas DataFrame object
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
    output_data.append([id, group.iloc[0]['Name']] + time_lag_pairs.flatten().tolist())

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
    nodes = []
    links = {'source': [], 'target': [], 'value': [], 'label': [], 'customdata': [], 'color':[]}

    colors = list(mcolors.CSS4_COLORS.keys())  # Get a list of available color names
    color_dict = {}  # Create an empty dictionary to store unique colors for each path

    for user in user_paths:
        path_str = '-'.join(user['path'])

        if path_str not in color_dict.keys():  # If this path doesn't have a unique color assigned
            color_dict[path_str] = colors.pop()  # Assign and remove one from the list

        for i in range(len(user['path']) - 1):
            if user['path'][i] not in nodes:
                nodes.append(user['path'][i])

            if user['path'][i + 1] not in nodes:
                nodes.append(user['path'][i + 1])

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
                x=0.95, y=(1 - 0.1 * i),  # Positions to stack annotations nicely
                showarrow=False
            )
        )
    layout = go.Layout(title_text="Sankey Diagram", font_size=10, annotations=annotations)
    fig = go.Figure(data=data, layout=layout)


    fig.show()




##########################################################################################################
# Create a new DataFrame from the rows containing 'summary'
purchase_data = df[df['Event'] == 'summary'].copy()
geolocator = Nominatim(user_agent="geoapiExercises")

class Application:
    def __init__(self, master, geolocator, purchase_data):
        self.master = master
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

        # Control variable for the End In mode
        self.end_in_mode = False

        # Geolocate the purchase data
        purchase_data['Latitude'], purchase_data['Longitude'] = zip(
            *purchase_data.apply(lambda row: self.geolocate(row['Country'], row['Region'], row['Mailing Address']), axis=1))

        # Count the frequency of each location
        self.location_counts = purchase_data.groupby(['Latitude', 'Longitude']).size().reset_index(name='Counts')

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

    def geolocate(self, country, region, address):
        # Combine the address, region, and country
        location_str = f"{address}, {region}, {country}"

        # Geolocate the combined string
        location = self.geolocator.geocode(location_str)

        if location != None:
            return location.latitude, location.longitude
        else:
            return np.nan, np.nan  # return NaN if location can't be found

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
        m = folium.Map(location=location, zoom_start=2)

        # Add a heatmap to the map
        HeatMap(data=self.location_counts[['Latitude', 'Longitude', 'Counts']].dropna(), radius=15).add_to(m)

        # Convert the map to HTML
        html = m._repr_html_()

        # Convert the HTML to an image
        with BytesIO() as file:
            m.save(file, close_file=False)
            image = base64.b64encode(file.getvalue()).decode('utf-8')

        # Create a new tkinter window
        new_window = tk.Toplevel()

        # Load the image into the window

        photo = tk.PhotoImage(data=image)
        label = tk.Label(new_window, image=photo)
        label.pack()

root = tk.Tk()
app = Application(root, geolocator, purchase_data)
root.mainloop()