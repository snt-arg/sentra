#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ament_index_python import get_package_share_directory
from sentra_ros.core.utils import cleanMemory, monitorParams


class Sentra(Node):
    def __init__(self):
        # Variables
        super().__init__(
            "sentra_ros",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.pkg_share_directory = get_package_share_directory("sentra_ros")
        # Initial checks
        monitorParams(self.get_logger())
        cleanMemory(self.get_logger())
        # Run core loop
        self.timer = self.create_timer(1.0, self.process)

    def process(self):
        # Main loop
        self.get_logger().info("Running core loop...")


def main(args=None):
    print("*** Sentra Started! ***")
    # Variables
    node = None
    rclpy.init(args=args)
    # Run the node
    try:
        node = Sentra()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[Sentra] Node interrupted by user! Exiting...")
    except Exception as e:
        node.get_logger().error(f"[Sentra] Unhandled exception: {e}")
    finally:
        if node is not None:
            node.destroy_node()


if __name__ == "__main__":
    main()
