"""Phase D task table: 9 tasks across three groups (baseline / rule gate /
learned VoI gate).  Definitions reuse the LightWM runtime semantics
(success conditions match scripts/ai2thor/work/run_task.py)."""

TASKS = [
    {
        "id": "potato_plate_microwave",
        "scene": "FloorPlan1",
        "description": ("Find the Potato, pick it up, put it on the Plate, "
                        "then put the Plate (with the Potato) into the "
                        "Microwave and close the microwave door."),
        "target_object_types": ["Potato", "Plate", "Microwave"],
        "success_conditions": [
            {"type": "object_state", "field": "parentReceptacles",
             "value": "Plate", "object_type": "Potato"},
            {"type": "object_state", "field": "parentReceptacles",
             "value": "Microwave", "object_type": "Plate"},
        ],
        "max_steps": 50,
    },
    {
        "id": "egg_pot",
        "scene": "FloorPlan2",
        "description": "Find the Egg, pick it up, and put it into the Pot.",
        "target_object_types": ["Egg", "Pot"],
        "success_conditions": [
            {"type": "object_state", "field": "parentReceptacles",
             "value": "Pot", "object_type": "Egg"},
        ],
        "max_steps": 50,
    },
    {
        "id": "keys_box",
        "scene": "FloorPlan3",
        "description": ("Find the KeyChain and the CreditCard, put them "
                        "both into the Box."),
        "target_object_types": ["KeyChain", "CreditCard", "Box"],
        "success_conditions": [
            {"type": "object_in_receptacle", "object_type": "KeyChain",
             "receptacle_type": "Box"},
            {"type": "object_in_receptacle", "object_type": "CreditCard",
             "receptacle_type": "Box"},
        ],
        "success_logic": "AND",
        "max_steps": 60,
    },
    {
        "id": "put_egg_in_pan",
        "scene": "FloorPlan2",
        "description": "Find the Egg, pick it up, and put it into the Pan.",
        "target_object_types": ["Egg", "Pan"],
        "success_conditions": [
            {"type": "object_state", "field": "parentReceptacles",
             "value": "Pan", "object_type": "Egg"},
        ],
        "max_steps": 50,
    },
    {
        "id": "slice_apple",
        "scene": "FloorPlan1",
        "description": "Find the Apple and slice it.",
        "target_object_types": ["Apple"],
        "success_conditions": [
            {"type": "object_state", "field": "isSliced",
             "value": True, "object_type": "Apple"},
        ],
        "max_steps": 60,
    },
    {
        "id": "open_fridge",
        "scene": "FloorPlan1",
        "description": ("Find the Fridge in the kitchen, walk to the front "
                        "of it, and open the fridge door."),
        "target_object_types": ["Fridge"],
        "success_conditions": [
            {"type": "object_state", "field": "isOpen",
             "value": True, "any": True},
        ],
        "max_steps": 40,
    },
    {
        "id": "open_microwave",
        "scene": "FloorPlan1",
        "description": ("Find the Microwave in the kitchen, walk to the "
                        "front of it, and open the microwave door."),
        "target_object_types": ["Microwave"],
        "success_conditions": [
            {"type": "object_state", "field": "isOpen",
             "value": True, "any": True},
        ],
        "max_steps": 40,
    },
    {
        "id": "turn_on_light",
        "scene": "FloorPlan10",
        "description": ("Find the light switch in the room, walk to the "
                        "front of it, and turn on the light."),
        "target_object_types": ["LightSwitch"],
        "success_conditions": [
            {"type": "object_state", "field": "isToggled",
             "value": True, "any": True},
        ],
        "max_steps": 40,
    },
    {
        "id": "pickup_phone",
        "scene": "FloorPlan2",
        "description": "Find the Phone, approach it, and pick it up.",
        "target_object_types": ["Phone"],
        "success_conditions": [
            {"type": "object_in_hand", "object_type": "Phone"},
        ],
        "max_steps": 30,
    },
]

GROUPS = {
    "A_baseline": {"memory_probe": False, "gate": "none"},
    "B_rule_gate": {"memory_probe": True, "gate": "rule"},
    "C_voi_gate": {"memory_probe": True, "gate": "learned"},
}
