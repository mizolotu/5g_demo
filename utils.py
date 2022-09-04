import numpy as np

def set_queues_cmd(port_name, qos_idx, rate_min=0.1e6, rate_max=10.1e6, rate_step=0.1e6):
    cmd = f'sudo ovs-vsctl set port {port_name} qos=@qos{qos_idx} -- --id=@qos{qos_idx} create qos type=linux-htb'
    for i, rate in enumerate(np.arange(rate_min, rate_max, rate_step)):
        cmd = f'{cmd} queues:{i}=@queue{i}'
    for i, rate in enumerate(np.arange(rate_min, rate_max, rate_step)):
        cmd = f'{cmd} -- --id=@queue{i} create queue other-config:max-rate={int(rate)}'
    return cmd

def generate_ovs_restart_file(controller_ip, iface1, oface2):

    lines =  [
        'sudo ovs-ofctl del-flows br',
        'sudo ovs-vsctl del-br br',
        'sudo ovs-vsctl -- --all destroy QoS -- --all destroy Queue',
        f'sudo ifconfig {iface1} 0',
        f'sudo ifconfig {iface2} 0',
        'sudo ovs-vsctl add-br br',
        f'sudo ovs-vsctl add-port br {iface1}',
        f'sudo ovs-vsctl add-port br {iface2}',
        f'sudo ovs-vsctl set-controller br tcp:{controller_ip}:6653',
        set_queues_cmd(iface1, 1),
        set_queues_cmd(iface2, 2)
    ]
    lines = '\n'.join(lines)
    with open('ovs_restart.sh', 'w') as f:
        f.writelines(lines)

    return lines

if __name__ == '__main__':

    controller_ip = '192.168.1.190'
    iface1 = 'enp0s8'
    iface2 = 'enp0s9'
    generate_ovs_restart_file(controller_ip, iface1, iface2)



