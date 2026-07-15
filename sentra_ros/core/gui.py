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
                        # Text Queries Section
                        dpg.add_text(
                            "Text Embeddings Registry:", color=ui_colors["orange"]
                        )
                        with dpg.child_window(height=200, tag="text_embeddings_panel"):
                            pass

                        dpg.add_spacer(height=10)

                        # Visual ROS Keyframes Section
                        dpg.add_text(
                            "Visual Embeddings Registry:", color=ui_colors["orange"]
                        )
                        with dpg.child_window(
                            height=200, tag="visual_embeddings_panel"
                        ):
                            pass
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

    def update_embeddings_tables(self):
        """
        Rebuilds both the Text Query registry table and the Visual rosbag keyframe table
        inside their respective scroll panels with current dataframe values.
        """
        # Render Text Embeddings Table
        if dpg.does_item_exist("text_embeddings_table"):
            dpg.delete_item("text_embeddings_table")

        text_df = self.node.query_text_df

        if text_df.empty:
            with dpg.group(parent="text_embeddings_panel", tag="text_embeddings_table"):
                dpg.add_text(
                    "No cached query text embeddings found yet.", color=[120, 120, 120]
                )
        else:
            with dpg.table(
                resizable=True,
                header_row=True,
                borders_outerH=True,
                borders_innerV=True,
                borders_innerH=True,
                tag="text_embeddings_table",
                parent="text_embeddings_panel",
                policy=dpg.mvTable_SizingStretchProp,
            ):
                dpg.add_table_column(label="Query Text", init_width_or_weight=0.4)
                dpg.add_table_column(
                    label="Embedding Stats (SigLIP)", init_width_or_weight=0.6
                )

                for index, row in text_df.iterrows():
                    with dpg.table_row():
                        dpg.add_text(str(row["query"]), wrap=180)

                        vector = row["embedding"]
                        if isinstance(vector, list) and len(vector) > 0:
                            # Safely flatten the list if it is nested
                            flat_vector = (
                                vector[0] if isinstance(vector[0], list) else vector
                            )
                            float_vector = [float(v) for v in flat_vector]

                            # Calculate stats
                            v_mean = sum(float_vector) / len(float_vector)
                            v_min = min(float_vector)
                            v_max = max(float_vector)

                            stats_text = f"Mean: {v_mean:+.4f}\nRange: [{v_min:.3f} to {v_max:.3f}] ({len(float_vector)} dims)"
                            dpg.add_text(stats_text, wrap=280, color=[150, 200, 255])
                        else:
                            dpg.add_text("Empty Vector", color=[120, 120, 120])

        # Render Visual Embeddings Table
        if dpg.does_item_exist("visual_embeddings_table"):
            dpg.delete_item("visual_embeddings_table")

        visual_df = self.node.kf_visual_df

        if visual_df.empty:
            with dpg.group(
                parent="visual_embeddings_panel", tag="visual_embeddings_table"
            ):
                dpg.add_text(
                    "No keyframe visual embeddings captured yet.", color=[120, 120, 120]
                )
        else:
            with dpg.table(
                resizable=True,
                header_row=True,
                borders_outerH=True,
                borders_innerV=True,
                borders_innerH=True,
                tag="visual_embeddings_table",
                parent="visual_embeddings_panel",
                policy=dpg.mvTable_SizingStretchProp,
            ):
                dpg.add_table_column(label="Frame", init_width_or_weight=0.1)
                dpg.add_table_column(label="Timestamp", init_width_or_weight=0.3)
                dpg.add_table_column(
                    label="Embedding Stats (Visual)", init_width_or_weight=0.6
                )

                for index, row in visual_df.iterrows():
                    with dpg.table_row():
                        # Format node ID and frame stamp
                        dpg.add_text(row["kf_id"], wrap=90)
                        dpg.add_text(row["timestamp"], wrap=90)

                        vector = row["embedding"]
                        if isinstance(vector, list) and len(vector) > 0:
                            # Safely flatten the list if it is nested
                            flat_vector = (
                                vector[0] if isinstance(vector[0], list) else vector
                            )
                            float_vector = [float(v) for v in flat_vector]

                            # Calculate stats
                            v_mean = sum(float_vector) / len(float_vector)
                            v_min = min(float_vector)
                            v_max = max(float_vector)

                            stats_text = f"Mean: {v_mean:+.4f}\nRange: [{v_min:.3f} to {v_max:.3f}] ({len(float_vector)} dims)"
                            dpg.add_text(stats_text, wrap=280, color=[180, 255, 180])
                        else:
                            dpg.add_text("Empty Vector", color=[120, 120, 120])
                        dpg.add_text("Empty Vector", color=[120, 120, 120])
