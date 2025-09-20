import getpass
import os
import json
import cohere


# Sample component database that can be fetched via tool calls
COMPONENT_DATABASE = {
    "components": {
        "Arduino Nano": {
            "type": "microcontroller",
            "pins": {
                "digital": ["D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13"],
                "analog": ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"],
                "power": ["+5V", "3.3V", "GND", "VIN"]
            },
            "specifications": {
                "operating_voltage": "5V",
                "input_voltage": "7-12V",
                "digital_pins": 14,
                "analog_pins": 8
            }
        },
        "L298N Motor Driver": {
            "type": "motor_driver",
            "pins": {
                "motor_a": ["OUT1", "OUT2"],
                "motor_b": ["OUT3", "OUT4"],
                "control": ["IN1", "IN2", "IN3", "IN4", "ENA", "ENB"],
                "power": ["+12V", "+5V", "GND"]
            },
            "specifications": {
                "max_current": "2A",
                "logic_voltage": "5V",
                "motor_voltage": "5V-35V"
            }
        },
        "DC Motor": {
            "type": "actuator",
            "pins": {
                "terminals": ["+", "-"]
            },
            "specifications": {
                "voltage": "3V-12V",
                "current": "0.1A-2A"
            }
        },
        "9V Battery": {
            "type": "power_source",
            "pins": {
                "terminals": ["+", "-"]
            },
            "specifications": {
                "voltage": "9V",
                "capacity": "500mAh-1000mAh"
            }
        }
    },
    "project_templates": {
        "robot_car": {
            "difficulty": "intermediate",
            "time_estimate": "2-3 hours",
            "required_components": ["Arduino Nano", "L298N Motor Driver", "DC Motor", "9V Battery"]
        }
    }
}


# Tool functions for Cohere to call
def get_component_info(component_name):
    """Get detailed information about a specific component."""
    if component_name in COMPONENT_DATABASE["components"]:
        return json.dumps(COMPONENT_DATABASE["components"][component_name])
    else:
        return json.dumps({"error": f"Component '{component_name}' not found in database"})

def get_all_components():
    """Get list of all available components."""
    return json.dumps(list(COMPONENT_DATABASE["components"].keys()))

def get_project_templates():
    """Get available project templates."""
    return json.dumps(COMPONENT_DATABASE["project_templates"])

# Tool definitions for Cohere
TOOLS = [
    {
        "name": "get_component_info",
        "description": "Get detailed technical information about a specific electronic component including pins, specifications, and connection details.",
        "parameter_definitions": {
            "component_name": {
                "description": "The name of the component to get information about",
                "type": "str",
                "required": True
            }
        }
    },
    {
        "name": "get_all_components", 
        "description": "Get a list of all available components in the database.",
        "parameter_definitions": {}
    },
    {
        "name": "get_project_templates",
        "description": "Get available project templates with difficulty levels and component requirements.",
        "parameter_definitions": {}
    }
]


# objects_using = ["Arduino Nano", "L298N Motor Driver", "2 DC Motor", "9V Battery"]

def get_query(objects_using):
    return '''
    Given ONLY the following materials: ''' + ', '.join(objects_using) + '''.
    Return a JSON object of the best project to work on with the name, description, instruction, connection, and components. 
    '''

def get_project_info_cohere(objects_using, api_key):
    '''get_project_info_cohere(objects_using, api_key) -> JSON
    returns the name description and instruction of the project using Cohere CommandA model with tool calls.
    '''

    # Initialize Cohere client V2
    co = cohere.ClientV2(api_key=api_key)
    
    personality = 'Have a fun and joking personality, like a comedian, but also detailed and easy to understand in explaining concepts.'
    message_format = '''
    Return the name, description, numbered steps instruction (in a fun way), connection, code, and components in the following JSON format:
    {
        "name": "xxx",
        "description": "xxx", 
        "instruction": "xxx",
        "connections": "xxx",
        "components": "xxx",
        "code": "xxx"
    }
    
    The components value will be an array of all the components used for the build. Multiple of the same component would each take an index.
    Connections will be an array data type not numbered showing the connections, in the exact format of "component;(connector name)$component;(connector name)", with each wire connection separate. 
    The Component is the same name as the input component name, unless there is duplicate then just add a number behind the component name.

    The wire position, connector name, and name of object will be specific and in short form (use symbols when applicable, ex positive to +) (Use short form for connector name when applicable, ex Digital Pin 5 to D5) (Use common labeling convention).
    In instruction refer to all the wire connections in the same format as the connection section.
    The instructions will be an array data type, with each index being a step.
    Use the available tools to get component information and project templates to enhance your response.
    Output Must Be a pure JSON.
    '''

    query = get_query(objects_using)
    
    # Tool call function handler
    def handle_tool_call(tool_call):
        if tool_call.function.name == "get_component_info":
            args = json.loads(tool_call.function.arguments)
            return get_component_info(args["component_name"])
        elif tool_call.function.name == "get_all_components":
            return get_all_components()
        elif tool_call.function.name == "get_project_templates":
            return get_project_templates()
        else:
            return json.dumps({"error": "Unknown tool"})

    # Initial message with system prompt and user query
    messages = [
        {
            "role": "system", 
            "content": personality + message_format
        },
        {
            "role": "user",
            "content": query
        }
    ]

    try:
        # Make initial call to CommandA model
        response = co.chat(
            model="command-r-plus-08-2024",  # CommandA model
            messages=messages,
            tools=TOOLS,
            response_format={"type": "json_object"}
        )

        # Handle tool calls if any
        if response.message.tool_calls:
            # Add the assistant's message with tool calls
            messages.append({
                "role": "assistant",
                "content": response.message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function", 
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    } for tool_call in response.message.tool_calls
                ]
            })

            # Execute tool calls and add results
            for tool_call in response.message.tool_calls:
                tool_result = handle_tool_call(tool_call)
                messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call.id
                })

            # Get final response after tool calls
            response = co.chat(
                model="command-r-plus-08-2024",
                messages=messages,
                response_format={"type": "json_object"}
            )

        print("Raw response:", response.message.content)
        
        # Parse the JSON response
        resp1 = json.loads(response.message.content)
        
        # Process the connections (existing logic)
        all_components = []
        all_connections = []
        for connection in resp1["connections"]:
            connection_point1, connection_point2 = connection.split("$")

            name1, connector_name1 = connection_point1.split(";")
            name2, connector_name2 = connection_point2.split(";")

            if name1 not in all_components:
                all_components.append(name1)
                all_connections.append([])

            if name2 not in all_components:
                all_components.append(name2)
                all_connections.append([])

            combo_name = connector_name1 + "/" + connector_name2

            all_connections[all_components.index(name1)].append(combo_name)
            all_connections[all_components.index(name2)].append(combo_name)

        resp1["connections"] = all_connections
        resp1["components"] = all_components

        return resp1
        
    except Exception as e:
        print(f"Error calling Cohere API: {e}")
        return {"error": str(e)}
        return {"error": str(e)}


# Example usage:
# objects_using = ["Arduino Nano", "L298N Motor Driver", "DC Motor", "DC Motor", "9V Battery"]
# api_key = "your_cohere_api_key_here"
# response = get_project_info_cohere(objects_using, api_key)
# print(response)
