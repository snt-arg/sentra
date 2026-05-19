import dearpygui.dearpygui as dpg


class SentraGUI:
    def __init__(self, ros_node):
        """
        Receives the running ROS 2 node instance so the GUI
        can interact with ROS parameters, logging, and processing.
        """
        self.node = ros_node
        self.setup_gui()

    def setup_gui(self):
        dpg.create_context()

        # GUI Layout
        try:
            with dpg.window(
                label="Sentra Gen-AI Grounding Hub",
                width=500,
                height=500,
                tag="MainWindow",
            ):
                # Scrollable chat history area
                with dpg.child_window(height=450, tag="chat_history"):
                    dpg.add_text(
                        "[Sentra]: I am ready for your queries...",
                        color=[100, 200, 100],
                    )

                dpg.add_separator()

                # Input row
                with dpg.group(horizontal=True):
                    dpg.add_input_text(
                        hint="Type your query here...",
                        tag="user_input",
                        width=415,
                        indent=5,
                    )
                    dpg.add_button(label="Send", callback=self.on_submit, width=80)
        except Exception as e:
            print(f"[Error] Error setting up the GUI: {e}")
            return

        # Finalize and show the GUI
        dpg.create_viewport(title="Sentra Gen-AI Grounding Hub", width=520, height=500)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("MainWindow", True)

    def on_submit(self, sender, app_data=None, *args, **kwargs):
        query = dpg.get_value("user_input")
        if not query.strip():
            return

        # Display the User's query in the chat log
        dpg.add_text(f"[User]: {query}", parent="chat_history")
        dpg.set_value("user_input", "")

        # Route processing through the ROS node
        self.node.process_query(query, self)

    def append_response(self, sender_name, text, color=[100, 150, 255]):
        """Helper method so the ROS node can pass back AI responses cleanly"""
        dpg.add_text(f"[{sender_name}]: {text}", parent="chat_history", color=color)
        dpg.set_y_scroll("chat_history", dpg.get_y_scroll_max("chat_history"))
