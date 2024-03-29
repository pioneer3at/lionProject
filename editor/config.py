DATA_INFO_FILENAME = 'programData.json'
TRAIN_IMAGE_NAME = 'train_image.jpg'

TOTAL_QUAN_FOR_JUDGEMENT = 3
DETECTION_LIST = [False] * TOTAL_QUAN_FOR_JUDGEMENT
ACCEPTANCE_VALUE = TOTAL_QUAN_FOR_JUDGEMENT - 1

WIDTH = 1280
HEIGHT = 720
FR = 30

MAX_LEFT_COUNTER = 10
VIDEO_EXTENSION = ".avi"

DEFAULT_VIDEO_FILENAME = 'output.avi'
PROGRAM_CONFIG_TEMPLATE = [
    {
        'type': '1 Book + 1 Crayon Pack',
        'typeId': 'A',
        'programId': 1,
        'name': None,
        'entries':
        [
            {'name': 'Crayon Quantity', 'data': 0, 'object': None, 'valid': 0},
        ],
        'steps':
        [
            {'name': 'Choose convey area', 'valid': False, 'stepId': 1, 'color': 'red', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose book+crayon pack area', 'valid': False, 'stepId': 2, 'color': 'yellow', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': True, 'imageName': 'full.jpg'},
            {'name': 'Choose crayon pack color area', 'valid': False, 'stepId': 3, 'color': 'orange', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': True, 'imageName': 'crayon.jpg'},
        ],
        'combobox':
        [
            {'name': 'Choose crayon direction', 'data': None, 'object': None, 'labelObject': None, 'options': ['Vertical', 'Horizontal']},
        ]
    },

    {
        'type': 'UPG',
        'typeId': 'UPG',
        'programId': 2,
        'name': None,
        'entries':
        [
            {'name': 'LED Quantity', 'data': 0, 'object': None, 'labelObject': None, 'valid': 0},
        ],
        'steps':
        [
            {'name': 'Choose convey area', 'valid': False, 'stepId': 1, 'color': 'red', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose 10 ports area', 'valid': False, 'stepId': 2, 'color': 'yellow', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': True, 'imageName': '10ports.jpg'},
            {'name': 'Choose LED areas', 'entryName': 'LED Quantity', 'valid': False, 'stepId': 3, 'color': 'red', 'quantity': 0, 'data': [], 'button': None, 'saveImage': False},
        ]
    },

    {
        'type': '1 Book + 1 Crayon Pack - B',
        'typeId': 'B',
        'programId': 3,
        'name': None,
        'entries':
        [
            {'name': 'Crayon Quantity', 'data': 0, 'object': None, 'valid': 0},
        ],
        'steps':
        [
            {'name': 'Choose convey area', 'valid': False, 'stepId': 1, 'color': 'red', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose crayon pack color + book area', 'valid': False, 'stepId': 2, 'color': 'orange', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': True, 'imageName': 'crayon.jpg'},
            {'name': 'Choose crayon pack color area', 'valid': False, 'stepId': 3, 'color': 'green', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose crayon areas', 'entryName': 'Crayon Quantity', 'valid': False, 'stepId': 4, 'color': 'pink', 'quantity': 0, 'data': [], 'button': None, 'saveImage': False, 'collect': 'hsv'},

        ],
        'combobox':
        [
            {'name': 'Choose crayon direction', 'data': None, 'object': None, 'labelObject': None, 'options': ['Vertical', 'Horizontal']},
        ]
    },
]