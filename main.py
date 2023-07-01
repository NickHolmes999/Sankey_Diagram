import pandas as pd
import csv
from datetime import datetime
import tkinter as tk
from tkinter import *
from tkinter import StringVar, OptionMenu, Label, Button, Text
import webbrowser
import random
import plotly.graph_objects as go
from plotly.offline import plot




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
def retrieve_user_paths():
    grouped_df = df.groupby('ID')
    user_paths = [list(group['Event']) for _, group in grouped_df]
    return user_paths


def generate_nodes_links():
    nodes = []
    links = {'source': [], 'target': [], 'value': [], 'label': [], 'data': []}

    user_paths = retrieve_user_paths()

    for event_path in user_paths:
        path_str = '-'.join(event_path)

        for i in range(len(event_path) - 1):
            if event_path[i] not in nodes:
                nodes.append(event_path[i])

            if event_path[i + 1] not in nodes:
                nodes.append(event_path[i + 1])

            links['source'].append(nodes.index(event_path[i]))
            links['target'].append(nodes.index(event_path[i + 1]))
            links['value'].append(1)
            links['label'].append(path_str)

    # Count unique paths and store in "data"
    links['data'] = pd.Series(links['label']).value_counts().to_dict()

    return nodes, links


# Create a Sankey Diagram
def create_sankey():
    nodes, links = generate_nodes_links()

    # Generate a unique color for each unique label in the data
    unique_labels = set(links['label'])
    colors = [f'hsl({i / len(unique_labels) * 360}, 50%, 50%)' for i in range(len(unique_labels))]
    color_dict = dict(zip(unique_labels, colors))

    # Assign corresponding colors to each link
    link_colors = [color_dict[label] for label in links['label']]

    # Convert dictionary 'links['data']' to a list of strings 'Path: Count'
    customdata_list = [f"{path}: {count}" for path, count in links['data'].items()]

    data = go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color="blue",
            hovertemplate='ID: %{customdata}<br>Name: %{label}<extra></extra>',
            customdata=nodes  # Added ID as customdata
        ),
        link=dict(
            source=links['source'],
            target=links['target'],
            value=links['value'],
            color=link_colors,
            hovertemplate='<b>%{label}<b>',  # Added hover template
            customdata=customdata_list  # Add the unique path count as customdata
        )
    )

    # Add insight box
    insight_text = '<br>'.join(f'{path}: {count}' for path, count in links['data'].items())
    annotations = [
        go.layout.Annotation(
            showarrow=False,
            text=insight_text,
            xanchor='right',
            x=1,
            yanchor='top',
            y=1,
         ),
    ]

    layout = go.Layout(title="Sankey Diagram", font=dict(size=10), annotations=annotations)
    fig = go.Figure(data=data, layout=layout)
    fig.show()


print("Hello")

class Application:
    def __init__(self, master):
        self.master = master
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

        self.sankey_button = Button(self.master, text="Show Sankey Diagram", command=create_sankey)
        self.sankey_button.pack()

        # Control variable for the End In mode
        self.end_in_mode = False

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


root = tk.Tk()
app = Application(root)
root.mainloop()
















