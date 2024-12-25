import os
import ctypes
import enum
import typing
import gameState


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

        # Load the library
        self.__lib_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "game.so"
        )
        self.__lib_instance = ctypes.cdll.LoadLibrary(self.__lib_path)

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
