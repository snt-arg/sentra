#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from ament_index_python import get_package_share_directory
from sentra_ros.core.utils import cleanMemory, monitorParams


class Sentra(Node):
    def __init__(self):
        # Variables
        self.pkg_share_directory = get_package_share_directory("sentra_ros")
        super().__init__("sentra")
        # Initial checks
        monitorParams()
        cleanMemory()
    
    def core(self):
        # Main loop
        print("Running core loop...")


def main(args=None):
    print("*** Sentra Started! ***")
    # Variables
    node = None
    rclpy.init(args=args)
    # Run the node
    try:
        node = Sentra()
    #     rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            '[Sentra] Node interrupted by user! Exiting...')
    except Exception as e:
        rclpy.logging.get_logger().error(
            f'[Sentra] Unhandled exception: {e}')
    finally:
        if node is not None:
            node.destroy_node()


if __name__ == "__main__":
    main()
