"""
⚜️ Sentra ⚜️
------------
* SPDX-FileCopyrightText: 2023-2026 University of Luxembourg
* SPDX-License-Identifier: SDF26-0040
* © 2023-2026 University of Luxembourg
* Developed by: Ali Tourani at SnT/ARG
* Sentra is licensed under the GPL 3.0 License
* (Check LICENSE file for details)
"""

import dearpygui.dearpygui as dpg
from sentra_ros.core.utils import ui_colors


class SentraGUI:
    def __init__(self, ros_node):
        """
        Receives the running ROS 2 node instance so the GUI
        can interact with ROS parameters, logging, and processing.
        """
        self.node = ros_node
        self.setup_gui()

    def setup_gui(self):
        # Variables
        logger = self.node.get_logger()
        if logger:
            logger.info("Loading Sentra GUI...")

        dpg.create_context()

        # GUI Layout
        try:
            with dpg.window(
                label="Sentra Gen-AI Grounding Hub",
                width=500,
                height=500,
                tag="MainWindow",
                no_resize=True,
                no_move=True,
                no_collapse=True,
            ):
                with dpg.tab_bar(tag="tab_bar"):
                    # Tab 1: Chat Interface
                    with dpg.tab(label="Chat", tag="chat_tab"):
                        # Scrollable chat history area (reduced height slightly to fit tab bar spacing)
                        with dpg.child_window(height=400, tag="chat_history"):
                            dpg.add_text(
                                "[Sentra]: I am ready for your queries...",
                                color=ui_colors["orange"],
                            )

                        dpg.add_separator()

                        # Input row
                        with dpg.group(horizontal=True):
                            dpg.add_input_text(
                                hint="Type your query here...",
                                tag="user_input",
                                width=390,  # Adjusted to fit nicely within the 500px window width
                            )
                            dpg.add_button(
                                label="Send", callback=self.on_submit, width=80
                            )

                    # Tab 2: Embeddings Registry
                    with dpg.tab(label="Embeddings", tag="embeddings_tab"):
                        dpg.add_text(
                            "Cached Query Text Registry", color=ui_colors["orange"]
                        )
                        dpg.add_separator()

                        # Scrollable panel containing the tabular Pandas values
                        with dpg.child_window(
                            height=410, tag="embeddings_scroll_panel"
                        ):
                            # We will draw the table dynamically here
                            self.update_embeddings_table()
        except Exception as e:
            print(f"[Error] Error setting up the GUI: {e}")
            return

        # Finalize and show the GUI
        dpg.create_viewport(title="Sentra Gen-AI Grounding Hub", width=520, height=500)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("MainWindow", True)

        if logger:
            logger.info("Sentra GUI initialized and ready!")

    def on_submit(self, sender, app_data=None, *args, **kwargs):
        query = dpg.get_value("user_input")
        if not query.strip():
            return

        # Display the User's query in the chat log
        dpg.add_text(f"[Query]: {query}", parent="chat_history", wrap=480)
        dpg.set_value("user_input", "")

        # Route processing through the ROS node
        self.node.process_query(query, self)

    def append_response(self, sender, text, color=ui_colors["orange"]):
        """Helper method so the ROS node can pass back AI responses cleanly"""
        dpg.add_text(
            f"[{sender}]: {text}", parent="chat_history", wrap=480, color=color
        )
        dpg.set_y_scroll("chat_history", dpg.get_y_scroll_max("chat_history"))

    def update_embeddings_table(self):
        """
        Rebuilds the table inside the embeddings scroll panel with current dataframe values.
        """
        # Delete the old table if it exists to avoid duplication
        if dpg.does_item_exist("query_embeddings_table"):
            dpg.delete_item("query_embeddings_table")

        # Fetch the DataFrame from the ROS Node parent reference
        df = self.node.query_text_df

        # If the DataFrame is empty, show a placeholder message
        if df.empty:
            with dpg.group(
                parent="embeddings_scroll_panel", tag="query_embeddings_table"
            ):
                dpg.add_text(
                    "No cached query text embeddings found yet.", color=[120, 120, 120]
                )
            return

        # Build the fresh table
        with dpg.table(
            header_row=True,
            resizable=True,
            policy=dpg.mvTable_SizingStretchProp,
            borders_outerH=True,
            borders_innerV=True,
            borders_innerH=True,
            parent="embeddings_scroll_panel",
            tag="query_embeddings_table",
        ):
            # Column definitions
            dpg.add_table_column(label="Query", init_width_or_weight=0.4)
            dpg.add_table_column(
                label="Embedding Preview", init_width_or_weight=0.6
            )

            # Populate rows
            for index, row in df.iterrows():
                with dpg.table_row():
                    # Display the input text query
                    dpg.add_text(str(row["query"]), wrap=180)
                    
                    vector = row["embedding"]
                    # Safely flatten the list if it is nested
                    if isinstance(vector[0], list):
                        flat_vector = vector[0]
                    else:
                        flat_vector = vector

                    # Cast to float safely
                    float_vector = [float(v) for v in flat_vector]
                        
                    # Calculate text statistics
                    v_mean = sum(float_vector) / len(float_vector)
                    v_min = min(float_vector)
                    v_max = max(float_vector)
                    
                    # Display clean statistical fingerprint
                    stats_text = f"Mean: {v_mean:+.4f}, Range: [{v_min:.3f} to {v_max:.3f}]"
                    dpg.add_text(stats_text, wrap=280)
