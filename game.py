import os
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
        __parent_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)))

        # Change the directory to work with the config.json file
        __lib_directory = os.path.join(__parent_directory, "bin")
        os.chdir(__lib_directory)

        # Load the library
        __lib_path = os.path.join(
            __lib_directory,
            "libgame.so",
        )
        self.__lib_instance = ctypes.cdll.LoadLibrary(__lib_path)

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
