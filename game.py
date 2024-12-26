import os
import sys
import ctypes
import enum
import typing
from . import gameState


class Game:
    """Represents an instance of the game

    Attributes:
        state: The current game state
    """

    class Action(enum.Enum):
        """Possible actions

        Attributes:
            UP: Move up
            LEFT: Move left
            RIGHT: Move right
            DOWN: Move down
        """

        UP = b"w"
        LEFT = b"a"
        RIGHT = b"d"
        DOWN = b"s"

    def __init__(self) -> None:
        self.state = gameState.GameState()

        # The directory of this file
        parent_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)))

        # Change the directory to work with the config.json file
        lib_directory = os.path.join(parent_directory, "bin")
        os.chdir(lib_directory)

        # Load the library
        if sys.platform == "linux":
            lib_path = os.path.join(
                lib_directory,
                "libgame.so",
            )
            self.__lib_instance = ctypes.cdll.LoadLibrary(lib_path)
        elif sys.platform == "win32":
            lib_path = os.path.join(lib_directory, "libgame.dll")
            self.__lib_instance = ctypes.WinDLL(lib_path)
        else:
            raise RuntimeError("Your OS is not supported!")

        # Load the C functions
        self.__reset = ctypes.CFUNCTYPE(ctypes.c_void_p)(
            ("Reset", self.__lib_instance),
        )
        self.__update = ctypes.CFUNCTYPE(ctypes.c_void_p)(
            ("Update", self.__lib_instance),
        )
        self.__step = ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_byte), ctypes.c_int
        )(
            ("Step", self.__lib_instance),
        )
        self.__observe = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_byte))(
            ("Obs", self.__lib_instance),
        )

        # Set up the game
        self.reset()

    def __update_state(self) -> None:
        self.state.update(
            bytes(
                ctypes.cast(
                    self.__observe(), ctypes.POINTER(ctypes.c_byte * 159)
                ).contents
            )
        )

    def reset(self) -> None:
        """Reset the game"""
        self.__reset()
        self.__update_state()

    def update(self) -> None:
        """Simulates a game tick"""
        self.__update()
        self.__update_state()

    def step(self, actions: typing.List[Action]) -> None:
        """Perform an action (does not update the game tick)

        Args:
            actions: A list of actions to take
        """
        # Create the binary message
        byte_string = b"".join([action.value for action in actions])
        byte_data = ctypes.c_byte * len(byte_string)
        byte_pointer = byte_data(*byte_string)

        self.__step(byte_pointer, len(byte_string))
        self.__update_state()
