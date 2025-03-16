import datetime
import smtplib
import tkinter as tk
import webbrowser
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import pickle
import numpy as np
import json
import folium
import matplotlib.pyplot as plt
import squarify
import seaborn as sns
import plotly.express as px
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plotly.subplots import make_subplots
import plotly.io as pio
from tkinterweb import HtmlFrame  # Used for rendering Plotly charts

# Load the trained model and label encoders
with open("air_quality_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Create a tkinter window
root = tk.Tk()
root.title('Air Quality Prediction and Visualization')
root.geometry("1000x700")
root.configure(bg='#2c3e50')

# Global dictionaries to store loaded CSV files and displayed predictions
file_dict = {}
displayed_predictions = []

# Function to encode user inputs using the trained label encoders
def encode_input(value, column):
    if column in label_encoders:
        if value in label_encoders[column].classes_:
            return label_encoders[column].transform([value])[0]
        else:
            return None
    return value

# Function to load CSV file and add to sidebar listbox
def load_csv():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            file_dict[file_path] = df
            messagebox.showinfo("Success", "CSV file loaded successfully!")
            # Add file path to the sidebar if not already present
            if file_path not in sidebar_listbox.get(0, tk.END):
                sidebar_listbox.insert(tk.END, file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file: {e}")

# Function to generate visualizations based on the selected CSV file
def visualize_csv():
    selected_file = sidebar_listbox.get(tk.ACTIVE)
    if not selected_file:
        messagebox.showerror("Error", "Please select a CSV file from the sidebar.")
        return

    df = file_dict.get(selected_file)
    if df is None:
        messagebox.showerror("Error", "Unable to retrieve the selected CSV file.")
        return

    # List to collect interactive Plotly HTML snippets
    interactive_htmls = []

    try:
        # Create a new window for the dashboard
        dashboard_window = tk.Toplevel()
        dashboard_window.title("Air Quality Dashboard")
        dashboard_window.geometry("1200x800")  # Set the window size

        # Create a main frame and a canvas for scrolling
        main_frame = tk.Frame(dashboard_window)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ====================== 1) Map Visualization (Pollutant Concentration) ======================
        try:
            pollutant_types = df['Pollutant Type'].unique()
            pollutant_var = tk.StringVar()
            pollutant_var.set(pollutant_types[0])  # Default to first pollutant
            pollutant_dropdown = ttk.Combobox(scrollable_frame, textvariable=pollutant_var, values=pollutant_types)
            pollutant_dropdown.pack(pady=10)

            def update_map(*args):
                selected_pollutant = pollutant_var.get()
                location_data = df[df['Pollutant Type'] == selected_pollutant].groupby(['Emirate'])["Value"].mean().reset_index()

                # Load UAE GeoJSON file (ensure file exists)
                uae_geojson_path = "uae_shapefile.geojson"
                with open(uae_geojson_path, "r") as file:
                    uae_geojson = json.load(file)

                map_center = [24.5, 54.0]
                air_quality_map = folium.Map(location=map_center, zoom_start=6)
                folium.Choropleth(
                    geo_data=uae_geojson,
                    name="choropleth",
                    data=location_data,
                    columns=["Emirate", "Value"],
                    key_on="feature.properties.name",
                    fill_color="YlOrRd",
                    fill_opacity=0.7,
                    line_opacity=0.2,
                    legend_name=f"Avg {selected_pollutant} Concentration"
                ).add_to(air_quality_map)
                temp_map_file = "temp_map.html"
                air_quality_map.save(temp_map_file)
                webbrowser.open(temp_map_file)
                messagebox.showinfo("Map Visualization", "Map opened in your browser.")
            pollutant_var.trace_add("write", update_map)
            update_map()
        except Exception as e:
            map_label = tk.Label(scrollable_frame, text=f"Error in map visualization: {str(e)}", fg="red")
            map_label.pack()

        # ====================== 2) Line Graph (Pollutants Over Time with Filter) ======================
        line_chart_frame = tk.Frame(scrollable_frame, bg='#34495e')
        line_chart_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        filter_frame = tk.LabelFrame(line_chart_frame, text="Filter Pollutants for Line Chart", bg='#34495e', fg='white', font=("Helvetica", 12, "bold"))
        filter_frame.pack(fill=tk.X, padx=10, pady=5)
        filter_canvas = tk.Canvas(filter_frame, bg='#34495e', height=100)
        filter_scrollbar = ttk.Scrollbar(filter_frame, orient="vertical", command=filter_canvas.yview)
        filter_inner = tk.Frame(filter_canvas, bg='#34495e')
        filter_inner.bind("<Configure>", lambda e: filter_canvas.configure(scrollregion=filter_canvas.bbox("all")))
        filter_canvas.create_window((0, 0), window=filter_inner, anchor="nw")
        filter_canvas.configure(yscrollcommand=filter_scrollbar.set)
        filter_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        filter_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        all_pollutants_line_var = tk.BooleanVar(value=False)
        pollutant_line_vars = {}
        pollutant_types = sorted(df["Pollutant Type"].unique().tolist())

        def toggle_all_line(state):
            for var in pollutant_line_vars.values():
                var.set(state)
        all_cb = tk.Checkbutton(filter_inner, text="All", variable=all_pollutants_line_var,
                                  command=lambda: toggle_all_line(all_pollutants_line_var.get()),
                                  bg='#34495e', fg='white', selectcolor='#2c3e50', font=("Helvetica", 10))
        all_cb.pack(anchor="w")
        for pt in pollutant_types:
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(filter_inner, text=pt, variable=var,
                                bg='#34495e', fg='white', selectcolor='#2c3e50', font=("Helvetica", 10))
            cb.pack(anchor="w")
            pollutant_line_vars[pt] = var

        chart_display_frame = tk.Frame(line_chart_frame, bg='#34495e')
        chart_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        def update_line_chart():
            selected_pollutants = [pt for pt, var in pollutant_line_vars.items() if var.get()]
            if not selected_pollutants:
                selected_pollutants = pollutant_types
            fig, ax = plt.subplots(figsize=(10, 5))
            for pollutant in selected_pollutants:
                df_filtered = df[df['Pollutant Type'] == pollutant].groupby('Year')["Value"].mean().reset_index()
                ax.plot(df_filtered['Year'], df_filtered['Value'], marker='o', label=pollutant)
            ax.set_title("Average Pollutant Concentration Over Time")
            ax.set_xlabel("Year")
            ax.set_ylabel("Average Concentration")
            ax.legend()
            ax.grid(True)
            for widget in chart_display_frame.winfo_children():
                widget.destroy()
            canvas_plot = FigureCanvasTkAgg(fig, master=chart_display_frame)
            canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas_plot.draw()

            # Create interactive Plotly line chart
            try:
                fig_plotly_line = px.line(df, x="Year", y="Value", color="Pollutant Type",
                                          title="Interactive Average Pollutant Concentration Over Time",
                                          template="plotly_dark")
                fig_plotly_line.update_layout(xaxis_tickangle=-45)
                # Convert to HTML snippet and add to our list
                html_line = pio.to_html(fig_plotly_line, full_html=False, include_plotlyjs='cdn')
                interactive_htmls.append(html_line)
                fig_plotly_line.show()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate interactive line chart: {e}")
        update_line_button = tk.Button(filter_frame, text="Update Line Chart", command=update_line_chart,
                                       bg='#27ae60', fg='white', font=("Helvetica", 12, "bold"))
        update_line_button.pack(pady=5)
        update_line_chart()

        # ====================== 3) Bar Graph: Average Concentration by Emirate (Interactive Plotly) ======================
        try:
            avg_by_emirate = df.groupby("Emirate")["Value"].mean().reset_index()
            fig_bar = px.bar(avg_by_emirate, x="Emirate", y="Value",
                             title="Average Pollutant Concentration by Emirate",
                             labels={"Value": "Average Concentration"},
                             color="Value", color_continuous_scale="Blues")
            fig_bar.update_layout(xaxis_tickangle=-45)
            interactive_htmls.append(pio.to_html(fig_bar, full_html=False, include_plotlyjs='cdn'))
            fig_bar.show()
        except Exception as e:
            bar_label = tk.Label(scrollable_frame, text=f"Error in bar graph visualization: {str(e)}", fg="red")
            bar_label.pack()

        # ====================== 4) Treemap: Average Concentration by Station Name with Filters (Interactive Plotly) ======================
        treemap_filter_frame = tk.Frame(scrollable_frame, bg='#34495e')
        treemap_filter_frame.pack(pady=10, fill=tk.X)

        def toggle_all(var_dict, state):
            for var in var_dict.values():
                var.set(state)

        treemap_pollutant_frame = tk.LabelFrame(treemap_filter_frame, text="Filter by Pollutant Type", bg='#34495e', fg='white', font=("Helvetica", 12, "bold"))
        treemap_pollutant_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)
        treemap_pollutant_canvas = tk.Canvas(treemap_pollutant_frame, bg='#34495e', height=100)
        treemap_pollutant_scrollbar = ttk.Scrollbar(treemap_pollutant_frame, orient="vertical", command=treemap_pollutant_canvas.yview)
        treemap_pollutant_inner = tk.Frame(treemap_pollutant_canvas, bg='#34495e')
        treemap_pollutant_inner.bind("<Configure>", lambda e: treemap_pollutant_canvas.configure(scrollregion=treemap_pollutant_canvas.bbox("all")))
        treemap_pollutant_canvas.create_window((0,0), window=treemap_pollutant_inner, anchor="nw")
        treemap_pollutant_canvas.configure(yscrollcommand=treemap_pollutant_scrollbar.set)
        treemap_pollutant_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        treemap_pollutant_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        treemap_pollutant_vars = {}
        for pt in sorted(df["Pollutant Type"].unique().tolist()):
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(treemap_pollutant_inner, text=pt, variable=var,
                                bg='#34495e', fg='white', selectcolor='#2c3e50', font=("Helvetica", 10))
            cb.pack(anchor="w")
            treemap_pollutant_vars[pt] = var

        treemap_station_frame = tk.LabelFrame(treemap_filter_frame, text="Filter by Station Name", bg='#34495e', fg='white', font=("Helvetica", 12, "bold"))
        treemap_station_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)
        treemap_station_canvas = tk.Canvas(treemap_station_frame, bg='#34495e', height=100)
        treemap_station_scrollbar = ttk.Scrollbar(treemap_station_frame, orient="vertical", command=treemap_station_canvas.yview)
        treemap_station_inner = tk.Frame(treemap_station_canvas, bg='#34495e')
        treemap_station_inner.bind("<Configure>", lambda e: treemap_station_canvas.configure(scrollregion=treemap_station_canvas.bbox("all")))
        treemap_station_canvas.create_window((0,0), window=treemap_station_inner, anchor="nw")
        treemap_station_canvas.configure(yscrollcommand=treemap_station_scrollbar.set)
        treemap_station_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        treemap_station_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        treemap_station_vars = {}
        for sn in sorted(df["Station Name"].unique().tolist()):
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(treemap_station_inner, text=sn, variable=var,
                                bg='#34495e', fg='white', selectcolor='#2c3e50', font=("Helvetica", 10))
            cb.pack(anchor="w")
            treemap_station_vars[sn] = var

        update_treemap_button = tk.Button(treemap_filter_frame, text="Update Treemap",
                                          command=lambda: update_treemap(treemap_pollutant_vars, treemap_station_vars),
                                          bg='#27ae60', fg='white', font=("Helvetica", 12, "bold"))
        update_treemap_button.pack(side=tk.LEFT, padx=10, pady=5)

        treemap_frame = tk.Frame(scrollable_frame, bg='#34495e')
        treemap_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        def update_treemap(pollutant_vars, station_vars):
            selected_pollutants = [pt for pt, var in pollutant_vars.items() if var.get()]
            if not selected_pollutants:
                selected_pollutants = sorted(df["Pollutant Type"].unique().tolist())
            selected_stations = [sn for sn, var in station_vars.items() if var.get()]
            if not selected_stations:
                selected_stations = sorted(df["Station Name"].unique().tolist())
            filtered_df = df[(df["Pollutant Type"].isin(selected_pollutants)) &
                             (df["Station Name"].isin(selected_stations))]
            if filtered_df.empty:
                messagebox.showinfo("No Data", "No data available for the selected filters.")
                return
            pivot = filtered_df.pivot_table(values="Value", index="Station Name", columns="Pollutant Type", aggfunc="mean", fill_value=0).reset_index()
            pivot["Total"] = pivot.drop("Station Name", axis=1).sum(axis=1)
            pivot["Label"] = pivot.apply(lambda x: f"{x['Station Name']}\nTotal: {x['Total']:.1f}", axis=1)

            fig_tree = px.treemap(pivot, path=['Station Name'], values='Total',
                                  color=pivot["NO2"] if "NO2" in pivot.columns else 'Total',
                                  color_continuous_scale='Viridis',
                                  title="Interactive Treemap: Average Pollutant Concentration by Station")
            fig_tree.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            interactive_htmls.append(pio.to_html(fig_tree, full_html=False, include_plotlyjs='cdn'))
            fig_tree.show()

        update_treemap(treemap_pollutant_vars, treemap_station_vars)

        # ====================== 5) Dot Plot: Pollutant Type vs. Station Location Type ======================
        dot_plot_frame = tk.Frame(scrollable_frame, bg='#34495e')
        dot_plot_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        df_avg = df.groupby(["Station Location Type", "Pollutant Type"])["Value"].mean().reset_index()
        fig_dot, ax_dot = plt.subplots(figsize=(12, 6))
        sns.set_style("whitegrid")
        sns.scatterplot(data=df_avg, x="Station Location Type", y="Value",
                        hue="Pollutant Type", size="Value", sizes=(50, 500),
                        alpha=0.8, edgecolor="black", palette="coolwarm", ax=ax_dot)
        for i, row in df_avg.iterrows():
            ax_dot.text(row["Station Location Type"], row["Value"], row["Pollutant Type"],
                        fontsize=9, ha="center", va="bottom")
        ax_dot.set_title("Dot Plot: Pollutant Type vs. Station Location Type", fontsize=14)
        ax_dot.set_xlabel("Station Location Type", fontsize=12)
        ax_dot.set_ylabel("Average Pollutant Value", fontsize=12)
        ax_dot.tick_params(axis='x', rotation=45)
        ax_dot.grid(True, linestyle="--", alpha=0.5)
        canvas_dot = FigureCanvasTkAgg(fig_dot, master=dot_plot_frame)
        canvas_dot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas_dot.draw()
        fig_plotly_dot = px.scatter(df_avg, x="Station Location Type", y="Value",
                                    size="Value", color="Pollutant Type",
                                    hover_data=["Pollutant Type"],
                                    title="Interactive Dot Plot: Pollutant Type vs. Station Location Type",
                                    labels={"Value": "Average Pollutant Value"}, template="plotly_dark")
        fig_plotly_dot.update_layout(xaxis_tickangle=-45)
        interactive_htmls.append(pio.to_html(fig_plotly_dot, full_html=False, include_plotlyjs='cdn'))
        fig_plotly_dot.show()

        # ====================== 6) Density Chart: Distribution of Pollutant Values ======================
        try:
            fig_density = px.density_heatmap(df, x="Pollutant Type", y="Value",
                                             title="Density Heatmap: Distribution of Pollutant Values",
                                             template="plotly_dark",
                                             color_continuous_scale="Viridis")
            fig_density.update_layout(xaxis_tickangle=-45)
            interactive_htmls.append(pio.to_html(fig_density, full_html=False, include_plotlyjs='cdn'))
            fig_density.show()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate density chart: {e}")

        # Combine all interactive HTML snippets into one HTML document and open it immediately
        try:
            combined_html = """
            <html>
            <head>
                <meta charset="utf-8">
                <title>Interactive Visualizations</title>
            </head>
            <body>
            """
            for html_snippet in interactive_htmls:
                combined_html += html_snippet + "<br><hr><br>"
            combined_html += "</body></html>"
            with open("combined_visuals.html", "w", encoding="utf-8") as f:
                f.write(combined_html)
            # Open the combined HTML file immediately in the default browser
            webbrowser.open("combined_visuals.html")
            messagebox.showinfo("Interactive Visuals", "Combined interactive visualizations have been saved and opened in your browser.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save combined interactive visuals: {e}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to generate visualizations: {e}")
    finally:
        print("Visualization generation complete.")





def predict_for_specific_inputs(emirate, station_type, pollutant_type):
    try:
        # Encode categorical values
        encoded_emirate = encode_input(emirate, "Emirate")
        encoded_station_type = encode_input(station_type, "Station Location Type")
        encoded_pollutant = encode_input(pollutant_type, "Pollutant Type")

        if None in [encoded_emirate, encoded_station_type, encoded_pollutant]:
            messagebox.showerror("Error", "Invalid input values. Please check your inputs.")
            return None

        # Automatically set prediction for 10 years into the future
        future_year = datetime.datetime.now().year + 10

        # Prepare input data with the exact column names used during training:
        input_data = pd.DataFrame([[future_year, encoded_pollutant, 0, encoded_emirate, 0, encoded_station_type]],
                                  columns=["Year", "Pollutant Type", "Country", "Emirate", "Station Name", "Station Location Type"])
        # Predict
        predicted_value = model.predict(input_data)[0]

        return predicted_value
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")
        return None

# Function to trigger prediction and display the results
def predict():
    try:
        emirate = emirate_entry.get().strip()
        station_type = station_entry.get().strip()
        pollutant_type = pollutant_entry.get().strip()

        if not emirate or not station_type or not pollutant_type:
            messagebox.showerror("Error", "Please enter all input fields (Emirate, Station Location Type, Pollutant Type).")
            return

        # Predict for 10 years into the future
        future_year = datetime.datetime.now().year + 10
        predicted_value = predict_for_specific_inputs(emirate, station_type, pollutant_type)

        if predicted_value is not None:
            results_treeview.insert("", tk.END, values=(emirate, station_type, pollutant_type,f"{predicted_value:.2f}"))
            displayed_predictions.append({
                "Emirate": emirate,
                "Station Type": station_type,
                "Pollutant": pollutant_type,
                "Predicted Value": predicted_value
            })

            messagebox.showinfo("Prediction Result", f"Predicted Value for {future_year}: {predicted_value:.2f}")

            if predicted_value > 0.5:  # Adjust threshold as needed
                send_email_alert(predicted_value, emirate, station_type, pollutant_type)

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process data: {e}")







# Function to refresh and clear predictions from display
def refresh_display():
    for row in results_treeview.get_children():
        results_treeview.delete(row)
    displayed_predictions.clear()

# Function to save the predictions displayed
def save_predictions():
    try:
        if not displayed_predictions:
            messagebox.showerror("Error", "No predictions to save.")
            return
        predictions_df = pd.DataFrame(displayed_predictions)
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            predictions_df.to_csv(save_path, index=False)
            messagebox.showinfo("Success", "Predictions saved successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save predictions: {e}")

# Function to send an alert email if predicted concentration exceeds threshold
def send_email_alert(predicted_value, emirate, station_type, pollutant_type):
    sender_email = "syedulazam6000@gmail.com"
    receiver_email = "syedulazam6000@gmail.com"
    password = "luyz jwth hzes kcxz"  # Use a secure method for storing passwords

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Air Quality Alert: High Pollutant Concentration"

    body = (f"Alert!\n\nThe predicted concentration of {pollutant_type} at {station_type} in {emirate} "
            f"has exceeded the threshold. The predicted value is {predicted_value:.2f}.\n\nPlease take necessary actions.")
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Error: Unable to send email alert. {e}")

# GUI Layout

# Title
title_label = tk.Label(root, text="Air Quality Prediction System", font=("Helvetica", 20, "bold"), fg='#ecf0f1', bg='#2c3e50')
title_label.pack(pady=20)

# Top Frame for File Loading and Visualization Buttons
top_frame = tk.Frame(root, bg='#34495e', padx=20, pady=10)
top_frame.pack(pady=10, fill=tk.X)
load_csv_button = tk.Button(top_frame, text="Load CSV File", command=load_csv, bg='#2980b9', fg='white', font=("Helvetica", 12, "bold"))
load_csv_button.pack(side=tk.LEFT, padx=10)
visualize_button = tk.Button(top_frame, text="Visualize CSV Data", command=visualize_csv, bg='#27ae60', fg='white', font=("Helvetica", 12, "bold"))
visualize_button.pack(side=tk.LEFT, padx=10)

# Left Frame for Prediction Inputs
left_frame = tk.Frame(root, bg='#34495e', padx=20, pady=20)
left_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.Y)

tk.Label(left_frame, text="Enter Prediction Inputs", bg='#34495e', fg='white', font=("Helvetica", 14, "bold")).pack(pady=10)

tk.Label(left_frame, text="Emirate:", fg="white", bg="#34495e", font=("Helvetica", 12)).pack(pady=5)
emirate_entry = tk.Entry(left_frame, font=("Helvetica", 12))
emirate_entry.pack(pady=5)

tk.Label(left_frame, text="Station Location Type:", fg="white", bg="#34495e", font=("Helvetica", 12)).pack(pady=5)
station_entry = tk.Entry(left_frame, font=("Helvetica", 12))
station_entry.pack(pady=5)

tk.Label(left_frame, text="Pollutant Type:", fg="white", bg="#34495e", font=("Helvetica", 12)).pack(pady=5)
pollutant_entry = tk.Entry(left_frame, font=("Helvetica", 12))
pollutant_entry.pack(pady=5)

# REMOVE Year Input (Since it's now a constant future prediction)
# tk.Label(left_frame, text="Year:", fg="white", bg="#34495e", font=("Helvetica", 12)).pack(pady=5)
# year_entry = tk.Entry(left_frame, font=("Helvetica", 12))
# year_entry.pack(pady=5)

predict_button = tk.Button(left_frame, text="Predict", command=predict, bg="#27ae60", fg="white", font=("Helvetica", 12, "bold"))
predict_button.pack(pady=10)

refresh_button = tk.Button(left_frame, text="Refresh Display", command=refresh_display, bg='#f39c12', fg='white', font=("Helvetica", 12, "bold"))
refresh_button.pack(pady=5)

save_button = tk.Button(left_frame, text="Save Predictions", command=save_predictions, bg='#2980b9', fg='white', font=("Helvetica", 12, "bold"))
save_button.pack(pady=5)

# Right Frame for Sidebar (Loaded CSV Files) and Results Table
right_frame = tk.Frame(root, bg='#34495e', bd=2, relief="solid")
right_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH, expand=True)

sidebar_label = tk.Label(right_frame, text="Loaded CSV Files", bg='#34495e', fg='white', font=("Helvetica", 14, "bold"))
sidebar_label.pack(pady=10)

# Sidebar Listbox for CSV file paths with scrollbar
sidebar_frame = tk.Frame(right_frame, bg='#34495e')
sidebar_frame.pack(pady=5, padx=10, fill=tk.X)
sidebar_scrollbar = ttk.Scrollbar(sidebar_frame, orient="vertical")
sidebar_listbox = tk.Listbox(sidebar_frame, font=("Helvetica", 12), yscrollcommand=sidebar_scrollbar.set)
sidebar_scrollbar.config(command=sidebar_listbox.yview)
sidebar_scrollbar.pack(side="right", fill="y")
sidebar_listbox.pack(side="left", fill="x", expand=True)

results_label = tk.Label(right_frame, text="Prediction Results", bg='#34495e', fg='white', font=("Helvetica", 14, "bold"))
results_label.pack(pady=10)

# Frame for results table with vertical scrollbar
results_frame = tk.Frame(right_frame, bg='#34495e', bd=2, relief="solid")
results_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical")
results_scrollbar.pack(side="right", fill="y")
results_treeview = ttk.Treeview(results_frame, columns=("Emirate", "Station Type", "Pollutant", "Predicted Value"), show="headings", yscrollcommand=results_scrollbar.set)
for col in ("Emirate", "Station Type", "Pollutant", "Predicted Value"):
    results_treeview.heading(col, text=col)
results_treeview.pack(fill=tk.BOTH, expand=True)
results_scrollbar.config(command=results_treeview.yview)

root.mainloop()

