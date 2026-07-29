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

import os
import numpy as np
import dearpygui.dearpygui as dpg
from PIL import Image as PILImage
from sentra_ros.core.utils import ui_colors, timestamp_to_time


class SentraGUI:
    def __init__(self, ros_node):
        """
        Receives the running ROS 2 node instance so the GUI
        can interact with ROS parameters, logging, and processing.
        """
        self.node = ros_node
        self.node.gui = self
        self.loaded_textures = set()
        self.setup_gui()

    def setup_gui(self):
        # Variables
        logger = self.node.get_logger()
        if logger:
            logger.info("Loading Sentra GUI...")

        dpg.create_context()

        # Global texture registry for image display
        with dpg.texture_registry(tag="texture_registry", show=False):
            pass

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

                    # Tab 3: Album Section
                    with dpg.tab(label="KeyFrame Album", tag="album_tab"):
                        dpg.add_text(
                            "Captured Keyframes Output Folder:",
                            color=ui_colors["orange"],
                        )
                        with dpg.child_window(height=400, tag="album_panel"):
                            dpg.add_text(
                                "No images saved in folder yet.",
                                tag="empty_album_text",
                                color=[120, 120, 120],
                            )
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
        self.update_album_gallery()

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
                dpg.add_table_column(label="Rel. Time", init_width_or_weight=0.3)
                dpg.add_table_column(
                    label="Embedding Stats (Visual)", init_width_or_weight=0.6
                )

                for index, row in visual_df.iterrows():
                    with dpg.table_row():
                        # Format node ID and frame stamp
                        dpg.add_text(row["kf_id"], wrap=90)
                        dpg.add_text(timestamp_to_time(row["timestamp"]), wrap=90)

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

    def load_texture_from_file(self, image_path, texture_tag):
        """
        Loads JPEG image using PIL and converts it to
        DearPyGui float32 RGBA texture format (No OpenCV required).
        """
        try:
            # Load image via PIL and convert to RGBA
            with PILImage.open(image_path) as img:
                img = img.convert("RGBA")
                width, height = img.size
                # Flatten normalized float array [0.0, 1.0] for DPG
                data = (np.array(img, dtype=np.float32) / 255.0).flatten()
            if dpg.does_item_exist(texture_tag):
                dpg.delete_item(texture_tag)
            dpg.add_static_texture(
                width=width,
                height=height,
                default_value=data,
                tag=texture_tag,
                parent="texture_registry",
            )
            return True, width, height
        except Exception as e:
            if hasattr(self, "node") and self.node:
                self.node.get_logger().error(
                    f"Failed to load image texture '{image_path}': {e}"
                )
            return False, 0, 0

    def update_album_gallery(self):
        """
        Reads keyframe JPEG files from output directory and populates the Album tab.
        """
        output_dir = getattr(self.node, "output_dir", None)
        if not output_dir or not os.path.exists(output_dir):
            return

        # Fetch all JPEG images sorted by creation time/name
        image_files = sorted(
            [f for f in os.listdir(output_dir) if f.endswith((".jpeg", ".jpg"))]
        )

        if not image_files:
            return

        if dpg.does_item_exist("empty_album_text"):
            dpg.delete_item("empty_album_text")

        # Container group for keyframes grid
        if not dpg.does_item_exist("album_grid"):
            dpg.add_group(tag="album_grid", parent="album_panel")

        for img_file in image_files:
            kf_id = os.path.splitext(img_file)[0]
            texture_tag = f"tex_{kf_id}"
            card_tag = f"card_{kf_id}"

            # Only process newly discovered images
            if kf_id not in self.loaded_textures:
                image_path = os.path.join(output_dir, img_file)
                success, w, h = self.load_texture_from_file(image_path, texture_tag)

                if success:
                    self.loaded_textures.add(kf_id)
                    with dpg.group(parent="album_grid", tag=card_tag):
                        dpg.add_text(f"📷 {img_file}", color=ui_colors["orange"])
                        # Render image with standard preview size (160x120)
                        dpg.add_image(texture_tag, width=160, height=120)
                        dpg.add_separator()
                        dpg.add_spacer(height=5)
