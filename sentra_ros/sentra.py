#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import dearpygui.dearpygui as dpg
from sentra_ros.core.gui import SentraGUI
from sentra_ros.core.embedding import initLLMModel
from ament_index_python import get_package_share_directory
from sentra_ros.core.utils import cleanMemory, monitorParams


class Sentra(Node):
    def __init__(self):
        # Variables
        self.pkg_share_directory = get_package_share_directory("sentra_ros")
        super().__init__(
            "sentra_ros",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        # Load parameters
        init_check = self.get_parameter("init_check").get_parameter_value().bool_value
        self.llm_model = (
            self.get_parameter("llm_model").get_parameter_value().string_value
        )

        # Initial checks
        if init_check:
            monitorParams(self.get_logger())
            cleanMemory(self.get_logger())

        # Initialize LLM model
        initLLMModel(self.llm_model, self.get_logger())

    def process_query(self, query, gui_handle):
        self.get_logger().info(f"Processing query: {query}")

        # --- YOUR GENAI / VS-GRAPH LOGIC GOES HERE ---
        response = f"Processed '{query}' successfully against vS-Graph."

        # Send result back to the UI layout safely
        gui_handle.append_response("Sentra", response)


def main(args=None):
    print("*** Sentra Started! ***\n")

    # Variables
    node = None
    rclpy.init(args=args)

    # Run the node
    try:
        node = Sentra()
        # Instantiate the GUI
        gui = SentraGUI(ros_node=node)
        # Main execution loop
        while dpg.is_dearpygui_running():
            rclpy.spin_once(node, timeout_sec=0.01)
            dpg.render_dearpygui_frame()

    except KeyboardInterrupt:
        if node:
            node.get_logger().info("[Sentra] Node interrupted by user! Exiting...")
    except Exception as e:
        if node:
            node.get_logger().error(f"[Sentra] Unhandled exception: {e}")
    finally:
        dpg.destroy_context()
        if node is not None:
            node.destroy_node()
        print("\n*** Sentra Shutdown Complete! ***")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
