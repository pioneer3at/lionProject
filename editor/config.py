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

BACKGROUND_SUBTRACTION_MIN_AREA = 10000

CORRECTNESS_PERCENTAGE_70   = 70
CORRECTNESS_PERCENTAGE_30   = 30
CORRECTNESS_PERCENTAGE_10   = 10

DEFAULT_VIDEO_FILENAME = 'output.avi'
PROGRAM_CONFIG_TEMPLATE = [
    # {
    #     'type': '1 Book + 1 Crayon Pack',
    #     'typeId': 'A',
    #     'programId': 1,
    #     'name': None,
    #     'entries':
    #     [
    #         {'name': 'Crayon Quantity', 'data': 0, 'object': None, 'valid': 0},
    #     ],
    #     'steps':
    #     [
    #         {'name': 'Choose convey area', 'valid': False, 'stepId': 1, 'color': 'red', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
    #         {'name': 'Choose book+crayon pack area', 'valid': False, 'stepId': 2, 'color': 'yellow', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': True, 'imageName': 'full.jpg'},
    #         {'name': 'Choose crayon pack color area', 'valid': False, 'stepId': 3, 'color': 'orange', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': True, 'imageName': 'crayon.jpg'},
    #     ],
    #     'combobox':
    #     [
    #         {'name': 'Choose crayon direction', 'data': None, 'object': None, 'labelObject': None, 'options': ['Vertical', 'Horizontal']},
    #     ]
    # },

    # {
    #     'type': 'UPG',
    #     'typeId': 'UPG',
    #     'programId': 2,
    #     'name': None,
    #     'entries':
    #     [
    #         {'name': 'LED Quantity', 'data': 0, 'object': None, 'labelObject': None, 'valid': 0},
    #     ],
    #     'steps':
    #     [
    #         {'name': 'Choose convey area', 'valid': False, 'stepId': 1, 'color': 'red', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
    #         {'name': 'Choose 10 ports area', 'valid': False, 'stepId': 2, 'color': 'yellow', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': True, 'imageName': '10ports.jpg'},
    #         {'name': 'Choose LED areas', 'entryName': 'LED Quantity', 'valid': False, 'stepId': 3, 'color': 'red', 'quantity': 0, 'data': [], 'button': None, 'saveImage': False},
    #     ]
    # },

    {
        'type': '1 Book + 1 Crayon Pack - B',
        'typeId': 'B',
        'programId': 3,
        'name': None,
        'packQuantity': 1,
        'entries':
        [
            {'name': 'Crayon Quantity', 'packId': 1, 'data': 0, 'object': None, 'valid': 0},
        ],
        'steps':
        [
            {'name': 'Choose convey area', 'multiple': False,  'packId': 0, 'valid': False, 'stepId': 1, 'color': 'red', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose crayon pack color + book area', 'packId': 0, 'multiple': False, 'valid': False, 'stepId': 2, 'color': 'orange', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': True, 'imageName': 'crayon.jpg'},
            {'name': 'Choose crayon pack color area', 'multiple': False, 'packId': 1, 'valid': False, 'stepId': 3, 'color': 'green', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose crayon areas', 'multiple': True, 'packId': 1, 'entryName': 'Crayon Quantity', 'valid': False, 'stepId': 4, 'color': 'pink', 'quantity': 0, 'data': [], 'button': None, 'saveImage': False, 'collect': 'hsv'},

        ],
        'combobox':
        [
            {'name': 'Choose crayon direction', 'packId': 1, 'data': None, 'object': None, 'labelObject': None, 'options': ['Vertical', 'Horizontal']},
        ]
    },

    {
        'type': '1 Book + 3 Crayon Pack - C',
        'typeId': 'C',
        'programId': 6,
        'name': None,
        'packQuantity': 3,
        'entries':
        [
            {'name': 'Pack 1 Quantity', 'data': 0, 'packId': 1, 'object': None, 'valid': 0},
            {'name': 'Pack 2 Quantity', 'data': 0, 'packId': 2, 'object': None, 'valid': 0},
            {'name': 'Pack 3 Quantity', 'data': 0, 'packId': 3, 'object': None, 'valid': 0},
        ],
        'steps':
        [
            {'name': 'Choose convey area', 'multiple': False,  'packId': 0, 'valid': False, 'stepId': 1, 'color': 'red', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            
            {'name': 'Choose pack 1 region', 'multiple': False,  'packId': 1, 'valid': False, 'stepId': 2, 'color': 'orange', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': True, 'imageName': 'pack1.jpg'},
            {'name': 'Choose pack 1 color region', 'multiple': False, 'packId': 1, 'valid': False, 'stepId': 3, 'color': 'green', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose color areas 1', 'multiple': True, 'packId': 1, 'entryName': 'Pack 1 Quantity', 'valid': False, 'stepId': 4, 'color': 'pink', 'quantity': 0, 'data': [], 'button': None, 'saveImage': False, 'collect': 'hsv'},
            
            {'name': 'Choose pack 2 region', 'multiple': False,  'packId': 2, 'valid': False, 'stepId': 5, 'color': 'orange', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': True, 'imageName': 'pack2.jpg'},
            {'name': 'Choose pack 2', 'multiple': False, 'packId': 2,'valid': False, 'stepId': 6, 'color': 'green', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose color areas 2', 'multiple': True, 'packId': 2, 'entryName': 'Pack 2 Quantity', 'valid': False, 'stepId': 7, 'color': 'pink', 'quantity': 0, 'data': [], 'button': None, 'saveImage': False, 'collect': 'hsv'},
            
            {'name': 'Choose pack 3 region', 'multiple': False,  'packId': 3, 'valid': False, 'stepId': 8, 'color': 'orange', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': True, 'imageName': 'pack3.jpg'},           
            {'name': 'Choose pack 3', 'multiple': False, 'packId': 3,'valid': False, 'stepId': 9, 'color': 'green', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose color areas 3', 'multiple': True, 'packId': 3, 'entryName': 'Pack 3 Quantity', 'valid': False, 'stepId': 10, 'color': 'pink', 'quantity': 0, 'data': [], 'button': None, 'saveImage': False, 'collect': 'hsv'},
        ],
        'combobox':
        [
            {'name': 'Choose pack 1 direction', 'packId': 1, 'data': None, 'object': None, 'labelObject': None, 'options': ['Vertical', 'Horizontal']},
            {'name': 'Choose pack 2 direction', 'packId': 2, 'data': None, 'object': None, 'labelObject': None, 'options': ['Vertical', 'Horizontal']},
            {'name': 'Choose pack 3 direction', 'packId': 3, 'data': None, 'object': None, 'labelObject': None, 'options': ['Vertical', 'Horizontal']},
        ]
    },

    {
        'type': 'Debug - D',
        'typeId': 'D',
        'programId': 5,
        'name': None,
        'packQuantity': 1,
        'entries':
        [],
        'steps':
        [],
        'combobox':
        [],
    },

    {
        'type': '1 Book + 3 Crayon Pack - E',
        'typeId': 'E',
        'programId': 7,
        'name': None,
        'packQuantity': 3,
        'entries':
        [
            {'name': 'Pack 1 Quantity', 'data': 0, 'packId': 1, 'object': None, 'valid': 0},
            {'name': 'Pack 2 Quantity', 'data': 0, 'packId': 2, 'object': None, 'valid': 0},
            {'name': 'Pack 3 Quantity', 'data': 0, 'packId': 3, 'object': None, 'valid': 0},
        ],
        'steps':
        [
            {'name': 'Choose convey area', 'multiple': False,  'packId': 0, 'valid': False, 'stepId': 1, 'color': 'red', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose full region', 'multiple': False,  'packId': 0, 'valid': False, 'stepId': 2, 'color': 'orange', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': True, 'imageName': 'full.jpg'},
            
            {'name': 'Choose pack 1 color region', 'multiple': False, 'packId': 1, 'valid': False, 'stepId': 3, 'color': 'green', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose color areas 1', 'multiple': True, 'packId': 1, 'entryName': 'Pack 1 Quantity', 'valid': False, 'stepId': 4, 'color': 'pink', 'quantity': 0, 'data': [], 'button': None, 'saveImage': False, 'collect': 'hsv'},
            
            {'name': 'Choose pack 2', 'multiple': False, 'packId': 2,'valid': False, 'stepId': 5, 'color': 'green', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose color areas 2', 'multiple': True, 'packId': 2, 'entryName': 'Pack 2 Quantity', 'valid': False, 'stepId': 6, 'color': 'pink', 'quantity': 0, 'data': [], 'button': None, 'saveImage': False, 'collect': 'hsv'},
            
            {'name': 'Choose pack 3', 'multiple': False, 'packId': 3,'valid': False, 'stepId': 7, 'color': 'green', 'data': [0, 0, 0, 0], 'button': None, 'saveImage': False},
            {'name': 'Choose color areas 3', 'multiple': True, 'packId': 3, 'entryName': 'Pack 3 Quantity', 'valid': False, 'stepId': 8, 'color': 'pink', 'quantity': 0, 'data': [], 'button': None, 'saveImage': False, 'collect': 'hsv'},
        ],
        'combobox':
        [
            {'name': 'Choose pack 1 direction', 'packId': 1, 'data': None, 'object': None, 'labelObject': None, 'options': ['Vertical', 'Horizontal']},
            {'name': 'Choose pack 2 direction', 'packId': 2, 'data': None, 'object': None, 'labelObject': None, 'options': ['Vertical', 'Horizontal']},
            {'name': 'Choose pack 3 direction', 'packId': 3, 'data': None, 'object': None, 'labelObject': None, 'options': ['Vertical', 'Horizontal']},
        ]
    },
]
