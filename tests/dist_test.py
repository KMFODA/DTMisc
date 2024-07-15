import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm 
import netifaces
import os
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

def find_network_interface(target_ip):
    for interface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addrs:
            for addr_info in addrs[netifaces.AF_INET]:
                if addr_info['addr'] == target_ip:
                    return interface
    # raise ValueError(f"No interface found for IP {target_ip}")
    print(f"No interface found for IP {target_ip}")
    return None

def find_default_interface():
    # Get the default gateway details
    gws = netifaces.gateways()
    default_gateway = gws['default'][netifaces.AF_INET]  # AF_INET for IPv4
    interface = default_gateway[1]
    print(f"Default internet interface: {interface}")
    # Optionally, get the IP address of the default interface
    addrs = netifaces.ifaddresses(interface)
    ip_info = addrs[netifaces.AF_INET][0]
    ip_address = ip_info['addr']
    print(f"IP Address of default interface: {ip_address}")
    return interface

"""
Sets the network interface for Gloo (used in distributed operations) based on the external IP.

Args:
    external_ip (str): The external IP address.
"""
def set_gloo_socket_ifname(external_ip):
    ifname = find_network_interface(external_ip)
    print(f"IP: {external_ip} IFNAME: {ifname}")
    if ifname is not None:
        os.environ['GLOO_SOCKET_IFNAME'] = ifname
    else:
        os.environ['GLOO_SOCKET_IFNAME'] = find_default_interface()

model = AutoModelForCausalLM.from_pretrained("kmfoda/gpt2-200m")
model = model.to("cuda")

#set_gloo_socket_ifname("188.241.30.201")
os.environ['GLOO_PORT_RANGE'] = "8080:8080"
os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
print("Initializing process group...")
dist.init_process_group(
        init_method=f"tcp://154.20.254.95:50512",
        backend='gloo',
        rank=1,
        world_size=2,
       # timeout=timedelta(seconds=60)
    )

world_size = dist.get_world_size()
print(f"World size: {world_size}")
    
for param in tqdm(model.parameters()): # TODO Consider doing all_reduce on full gradient vector instead of loop
    if param.grad is not None:
        # Compress to FP16 # See: https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        #grad_fp16 = param.grad.data.to(torch.float16)
        print("Performing all_reduce..")
        # All-reduce: Sum the gradients across all nodes
        dist.all_reduce(grad_fp16, op=dist.ReduceOp.SUM)
        print("Finished all_reduce..")

        # Average the gradients
        grad_fp16 /= world_size
        
        # Decompress (convert back to FP32 or original dtype) see above link
        #param.grad.data = grad_fp16.to(param.grad.data.dtype)

dist.destroy_process_group()
print("Process group destroyed.")