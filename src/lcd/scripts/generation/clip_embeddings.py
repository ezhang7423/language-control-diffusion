# %%
import open_clip
from torchinfo import summary
from lcd import DATA_PATH
import torch
import json

model, _, preprocess = open_clip.create_model_and_transforms('RN50', pretrained='openai')
tokenizer = open_clip.get_tokenizer('RN50')
NAME = "clip-rn50"


summary(model)


# anns = sum(json.load(open(DATA_PATH /"annotations.json")).values(), [])
anns = ['grasp the red block, then rotate it right',
 'grasp the red block, then turn it right',
 'grasp the red block and rotate it right',
 'grasp the red block and turn it right',
 'take the red block and rotate it right',
 'take the red block and turn it right',
 'rotate right the red block',
 'rotate the red block 90 degrees to the right',
 'rotate the red block to the right',
 'rotate the red block towards the right',
 'turn the red block right',
 'grasp the red block, then rotate it left',
 'grasp the red block, then turn it left',
 'grasp the red block and rotate it left',
 'grasp the red block and turn it left',
 'take the red block and rotate it left',
 'take the red block and turn it left',
 'rotate the red block 90 degrees to the left',
 'rotate left the red block',
 'rotate the red block to the left',
 'rotate the red block towards the left',
 'turn the red block left',
 'grasp the blue block, then rotate it right',
 'grasp the blue block, then turn it right',
 'grasp the blue block and rotate it right',
 'grasp the blue block and turn it right',
 'take the blue block and rotate it right',
 'take the blue block and turn it right',
 'rotate the blue block 90 degrees to the right',
 'rotate right the blue block',
 'rotate the blue block to the right',
 'rotate the blue block towards the right',
 'turn the blue block right',
 'grasp the blue block, then rotate it left',
 'grasp the blue block, then turn it left',
 'grasp the blue block and rotate it left',
 'grasp the blue block and turn it left',
 'take the blue block and rotate it left',
 'take the blue block and turn it left',
 'rotate the blue block 90 degrees to the left',
 'rotate left the blue block',
 'rotate the blue block to the left',
 'rotate the blue block towards the left',
 'turn the blue block left',
 'grasp the pink block, then rotate it right',
 'grasp the pink block, then turn it right',
 'grasp the pink block and rotate it right',
 'grasp the pink block and turn it right',
 'take the pink block and rotate it right',
 'take the pink block and turn it right',
 'rotate the pink block 90 degrees to the right',
 'rotate right the pink block',
 'rotate the pink block to the right',
 'rotate the pink block towards the right',
 'turn the pink block right',
 'grasp the pink block, then rotate it left',
 'grasp the pink block, then turn it left',
 'grasp the pink block and rotate it left',
 'grasp the pink block and turn it left',
 'take the pink block and rotate it left',
 'take the pink block and turn it left',
 'rotate the pink block 90 degrees to the left',
 'rotate left the pink block',
 'rotate the pink block to the left',
 'rotate the pink block towards the left',
 'turn the pink block left',
 'push the red block towards the right',
 'push right the red block',
 'push the red block to the right',
 'go push the red block to the right',
 'slide the red block towards the right',
 'slide right the red block',
 'slide the red block to the right',
 'sweep the red block to the right',
 'go slide the red block to the right',
 'push the red block towards the left',
 'push left the red block',
 'push the red block to the left',
 'go push the red block to the left',
 'slide the red block towards the left',
 'slide left the red block',
 'slide the red block to the left',
 'sweep the red block to the left',
 'go slide the red block to the left',
 'push the blue block towards the right',
 'push right the blue block',
 'push the blue block to the right',
 'go push the blue block to the right',
 'slide the blue block towards the right',
 'slide right the blue block',
 'slide the blue block to the right',
 'sweep the blue block to the right',
 'go slide the blue block to the right',
 'push the blue block towards the left',
 'push left the blue block',
 'push the blue block to the left',
 'go push the blue block to the left',
 'slide the blue block towards the left',
 'slide left the blue block',
 'slide the blue block to the left',
 'sweep the blue block to the left',
 'go slide the blue block to the left',
 'push the pink block towards the right',
 'push right the pink block',
 'push the pink block to the right',
 'go push the pink block to the right',
 'slide the pink block towards the right',
 'slide right the pink block',
 'slide the pink block to the right',
 'sweep the pink block to the right',
 'go slide the pink block to the right',
 'push the pink block towards the left',
 'push left the pink block',
 'push the pink block to the left',
 'go push the pink block to the left',
 'slide the pink block towards the left',
 'slide left the pink block',
 'slide the pink block to the left',
 'sweep the pink block to the left',
 'go slide the pink block to the left',
 'grasp the door handle, then slide the door to the left',
 'grasp the door handle, then move the door to the left',
 'grasp the door handle and slide the door to the left',
 'grasp the door handle and move the door to the left',
 'move the door all the way to the left',
 'slide the door all the way to the left',
 'move the door to the left',
 'slide the door to the left',
 'push the door to the left',
 'move the door to the left side',
 'slide the door to the left side',
 'push the door to the left side',
 'slide the door to the left, then let it go',
 'move the door all the way to the left and let go',
 'move the sliding door to the left',
 'push the sliding door to the left',
 'grasp the door handle, then slide the door to the right',
 'grasp the door handle, then move the door to the right',
 'grasp the door handle and slide the door to the right',
 'grasp the door handle and move the door to the right',
 'move the door all the way to the right',
 'slide the door all the way to the right',
 'move the door to the right',
 'slide the door to the right',
 'push the door to the right',
 'move the door to the right side',
 'slide the door to the right side',
 'push the door to the right side',
 'slide the door to the right, then let it go',
 'move the door all the way to the right and let go',
 'move the sliding door to the right',
 'push the sliding door to the right',
 'grasp the drawer handle, then open it',
 'grasp the drawer handle and open it',
 'grasp the handle of the drawer, then open it',
 'grasp the handle of the drawer and open it',
 'open the drawer',
 'go open the drawer',
 'pull the handle of the drawer',
 'pull the drawer',
 'open the cabinet drawer',
 'grasp the drawer handle, then close it',
 'grasp the drawer handle and close it',
 'grasp the handle of the drawer, then close it',
 'grasp the handle of the drawer and close it',
 'close the drawer',
 'go close the drawer',
 'push the handle of the drawer',
 'push the drawer',
 'close the cabinet drawer',
 'lift the red block from the table',
 'pick up the red block on the table',
 'pick up the red block from the table',
 'lift the red block',
 'pick up the red block',
 'lift the red block up',
 'grasp the red block on the table and lift it up',
 'grasp the red block and lift it up',
 'grasp the red block on the table, then lift it up',
 'grasp the red block, then lift it up',
 'lift the blue block from the table',
 'pick up the blue block on the table',
 'pick up the blue block from the table',
 'lift the blue block',
 'pick up the blue block',
 'lift the blue block up',
 'grasp the blue block on the table and lift it up',
 'grasp the blue block and lift it up',
 'grasp the blue block on the table, then lift it up',
 'grasp the blue block, then lift it up',
 'lift the pink block from the table',
 'pick up the pink block on the table',
 'pick up the pink block from the table',
 'lift the pink block',
 'pick up the pink block',
 'lift the pink block up',
 'grasp the pink block on the table and lift it up',
 'grasp the pink block and lift it up',
 'grasp the pink block on the table, then lift it up',
 'grasp the pink block, then lift it up',
 'pick up the red block from the shelf',
 'pick up the red block from the sliding cabinet',
 'pick up the red block in the sliding cabinet',
 'grasp the red block lying on the shelf',
 'grasp the red block lying in the cabinet',
 'grasp the red block lying in the sliding cabinet',
 'grasp the red block lying in the slider',
 'lift the red block lying on the shelf',
 'lift the red block lying in the cabinet',
 'lift the red block lying in the sliding cabinet',
 'lift the red block lying in the slider',
 'in the slider pick up the red block',
 'in the cabinet pick up the red block',
 'in the slider grasp the red block',
 'in the cabinet grasp the red block',
 'in the sliding cabinet grasp the red block',
 'lift the red block on the shelf',
 'pick up the blue block from the shelf',
 'pick up the blue block from the sliding cabinet',
 'pick up the blue block in the sliding cabinet',
 'grasp the blue block lying on the shelf',
 'grasp the blue block lying in the cabinet',
 'grasp the blue block lying in the sliding cabinet',
 'grasp the blue block lying in the slider',
 'lift the blue block lying on the shelf',
 'lift the blue block lying in the cabinet',
 'lift the blue block lying in the sliding cabinet',
 'lift the blue block lying in the slider',
 'in the slider pick up the blue block',
 'in the cabinet pick up the blue block',
 'in the slider grasp the blue block',
 'in the cabinet grasp the blue block',
 'in the sliding cabinet grasp the blue block',
 'lift the blue block on the shelf',
 'pick up the pink block from the shelf',
 'pick up the pink block from the sliding cabinet',
 'pick up the pink block in the sliding cabinet',
 'grasp the pink block lying on the shelf',
 'grasp the pink block lying in the cabinet',
 'grasp the pink block lying in the sliding cabinet',
 'grasp the pink block lying in the slider',
 'lift the pink block lying on the shelf',
 'lift the pink block lying in the cabinet',
 'lift the pink block lying in the sliding cabinet',
 'lift the pink block lying in the slider',
 'in the slider pick up the pink block',
 'in the cabinet pick up the pink block',
 'in the slider grasp the pink block',
 'in the cabinet grasp the pink block',
 'in the sliding cabinet grasp the pink block',
 'lift the pink block on the shelf',
 'grasp the red block from the drawer',
 'grasp the red block lying in the drawer',
 'grasp the red block in the drawer',
 'pick up the red block lying in the drawer',
 'pick up the red block from the drawer',
 'pick up the red block in the drawer',
 'go towards the red block in the drawer and pick it up',
 'go towards the red block in the drawer and grasp it',
 'go towards the red block in the drawer and lift it',
 'lift the red block in the drawer',
 'lift the red block lying in the drawer',
 'grasp the blue block from the drawer',
 'grasp the blue block lying in the drawer',
 'grasp the blue block in the drawer',
 'pick up the blue block lying in the drawer',
 'pick up the blue block from the drawer',
 'pick up the blue block in the drawer',
 'go towards the blue block in the drawer and pick it up',
 'go towards the blue block in the drawer and grasp it',
 'go towards the blue block in the drawer and lift it',
 'lift the blue block in the drawer',
 'lift the blue block lying in the drawer',
 'grasp the pink block from the drawer',
 'grasp the pink block lying in the drawer',
 'grasp the pink block in the drawer',
 'pick up the pink block lying in the drawer',
 'pick up the pink block from the drawer',
 'pick up the pink block in the drawer',
 'go towards the pink block in the drawer and pick it up',
 'go towards the pink block in the drawer and grasp it',
 'go towards the pink block in the drawer and lift it',
 'lift the pink block in the drawer',
 'lift the pink block lying in the drawer',
 'place in slider',
 'put it in the slider',
 'place the block in the sliding cabinet',
 'place the object in the sliding cabinet',
 'place the grasped object in the sliding cabinet',
 'put the block in the sliding cabinet',
 'put the object in the sliding cabinet',
 'put the grasped object in the sliding cabinet',
 'place the block in the cabinet',
 'place the object in the cabinet',
 'place the grasped object in the cabinet',
 'put the block in the cabinet',
 'put the object in the cabinet',
 'put the grasped object in the cabinet',
 'place the block in the slider',
 'place the object in the slider',
 'place the grasped object in the slider',
 'put the block in the slider',
 'put the object in the slider',
 'put the grasped object in the slider',
 'place the block in the drawer',
 'place the object in the drawer',
 'place the grasped object in the drawer',
 'put the block in the drawer',
 'put the object in the drawer',
 'put the grasped object in the drawer',
 'store the block in the drawer',
 'store the object in the drawer',
 'store the grasped object in the drawer',
 'move to the drawer and place the object',
 'go towards the drawer and place the object',
 'move to the drawer, then place the object',
 'move to the drawer and store the object',
 'go towards the drawer and store the object',
 'move to the drawer, then store the object',
 'push the object into the drawer',
 'push the block into the drawer',
 'slide the object into the drawer',
 'slide the block into the drawer',
 'sweep the object into the drawer',
 'sweep the block into the drawer',
 'push the object that it falls into the drawer',
 'stack blocks on top of each other',
 'stack the blocks',
 'stack the object on top of another object',
 'place the block on top of another block',
 'place the grasped block on top of another block',
 'put the grasped block on top of a block',
 'put the block on top of another block',
 'stack the block on top of another block',
 'collapse the stacked blocks',
 'take off the stacked block',
 'unstack the blocks',
 'go to the tower of blocks and take off the top one',
 'remove a block from the stack',
 'take off the block that is on top of the other one',
 'remove the top block',
 'turn on the light bulb',
 'turn on the yellow light',
 'turn on the yellow lamp',
 'move up the switch',
 'push the switch upwards',
 'slide up the switch',
 'move the light switch to turn on the light bulb',
 'toggle the light switch to turn on the light bulb',
 'move the light switch to turn on the yellow light',
 'toggle the light switch to turn on the yellow light',
 'turn off the light bulb',
 'turn off the yellow light',
 'turn off the yellow lamp',
 'move down the switch',
 'push the switch downwards',
 'slide down the switch',
 'move the light switch to turn off the light bulb',
 'toggle the light switch to turn off the light bulb',
 'move the light switch to turn off the yellow light',
 'toggle the light switch to turn off the yellow light',
 'turn on the led light',
 'turn on the led',
 'turn on the led lamp',
 'turn on the green light',
 'turn on the green lamp',
 'push down the button to turn on the led light',
 'push down the button to turn on the led',
 'push down the button to turn on the green light',
 'push the button to turn on the led light',
 'push the button to turn on the led',
 'push the button to turn on the green light',
 'toggle the button to turn on the led light',
 'toggle the button to turn on the led',
 'toggle the button to turn on the green light',
 'turn off the led light',
 'turn off the led',
 'turn off the led lamp',
 'turn off the green light',
 'turn off the green lamp',
 'push down the button to turn off the led light',
 'push down the button to turn off the led',
 'push down the button to turn off the green light',
 'push the button to turn off the led light',
 'push the button to turn off the led',
 'push the button to turn off the green light',
 'toggle the button to turn off the led light',
 'toggle the button to turn off the led',
 'toggle the button to turn off the green light',
 'take the red block and rotate it to the right',
 'take the red block and rotate it to the left',
 'take the blue block and rotate it to the right',
 'take the blue block and rotate it to the left',
 'take the pink block and rotate it to the right',
 'take the pink block and rotate it to the left',
 'go push the red block right',
 'go push the red block left',
 'go push the blue block right',
 'go push the blue block left',
 'go push the pink block right',
 'go push the pink block left',
 'push the sliding door to the left side',
 'push the sliding door to the right side',
 'pull the handle to open the drawer',
 'push the handle to close the drawer',
 'grasp and lift the red block',
 'grasp and lift the blue block',
 'grasp and lift the pink block',
 'lift the red block from the sliding cabinet',
 'lift the blue block from the sliding cabinet',
 'lift the pink block from the sliding cabinet',
 'Take the red block from the drawer',
 'Take the blue block from the drawer',
 'Take the pink block from the drawer',
 'store the grasped block in the sliding cabinet',
 'store the grasped block in the drawer',
 'slide the block that it falls into the drawer',
 'stack the grasped block',
 'remove the stacked block',
 'use the switch to turn on the light bulb',
 'use the switch to turn off the light bulb',
 'press the button to turn on the led light',
 'press the button to turn off the led light']
embeds = {}
with torch.no_grad(), torch.cuda.amp.autocast():
    embeddings = model.encode_text(tokenizer(anns))
    
for a, e in zip(anns, embeddings):
    if len(e.shape) == 1:
        e = e[None]
    embeds[a] = e.cpu()

print('Saving')    
save_path = DATA_PATH/ f"{NAME}_embeddings.pt"
torch.save(embeds, save_path)
print(f"Saved to {save_path}")