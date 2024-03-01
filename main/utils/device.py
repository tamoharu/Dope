from typing import Any, List


def encode_devices(devices: List[str]) -> List[Any]:
    encoded_devices: List[Any] = []
    for device in devices:
        if isinstance(device, tuple):
            encoded_devices.append(device[0].replace('ExecutionProvider', '').lower())
        else:
            encoded_devices.append(device.replace('ExecutionProvider', '').lower())
    return encoded_devices


def decode_devices(devices: List[str]) -> List[Any]:
    decoded_devices: List[Any] = []
    for device in devices:
        decoded_execution_provider = device.upper() + 'ExecutionProvider'
        if device == 'cuda':
            decoded_devices.append((decoded_execution_provider, {'cudnn_conv_algo_search': 'DEFAULT'}))
        else:
            decoded_devices.append(decoded_execution_provider)
    return decoded_devices
