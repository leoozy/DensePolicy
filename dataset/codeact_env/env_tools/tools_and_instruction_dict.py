from dataset.codeact_env.env_tools.alfworld_tools import Get_ALFWORLD_TOOLS, ALFWORLD_TOOLS_Instruction
from dataset.codeact_env.env_tools.babyai_tools import Get_BABYAI_TOOLS, BABYAI_TOOLS_Instruction
from dataset.codeact_env.env_tools.maze_tools import MAZE_TOOLS_Instruction
from dataset.codeact_env.env_tools.webshop_tools import WEBSHOP_TOOLS_Instruction

tools_dict = {'alfworld': Get_ALFWORLD_TOOLS, 'babyai': Get_BABYAI_TOOLS}

tools_instruction_dict = {'alfworld': ALFWORLD_TOOLS_Instruction, 'babyai': BABYAI_TOOLS_Instruction,
                          'maze': MAZE_TOOLS_Instruction, 'webshop': WEBSHOP_TOOLS_Instruction}
