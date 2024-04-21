import random
from bitcoin import *

def generate_vanity_address(suffix, testnet=True):
    while True:
        # Generate a random private key
        private_key = random_key()
        
        # Derive the public key from the private key
        public_key = privtopub(private_key)
        
        # Derive the Bitcoin address (P2PKH) from the public key
        address = pubtoaddr(public_key, magicbyte=111 if testnet else 0)
        
        # Check if the address ends with the desired suffix (case-insensitive)
        if address.lower().endswith(suffix.lower()):
            return private_key, address

# Specify the desired suffix (first 5 letters of your name)
suffix = "dacot"

# Generate the vanity address
private_key, address = generate_vanity_address(suffix)

print(f"Vanity Address: {address}")
print(f"Private Key: {private_key}")