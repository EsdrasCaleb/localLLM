def load_env_file(file_path):
    env_dict = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Remove leading/trailing whitespace and newline characters
                line = line.strip()

                # Ignore comments and empty lines
                if line and not line.startswith("#"):
                    key_value = line.split("=", 1)

                    # Ensure there are exactly two parts: key and value
                    if len(key_value) == 2:
                        key, value = key_value
                        key = key.strip()
                        value = value.strip()

                        # Optionally, interpret booleans and numbers
                        if value.lower() in ["true", "false"]:
                            value = value.lower() == "true"
                        elif value.isdigit():
                            value = int(value)

                        env_dict[key] = value

    except FileNotFoundError:
        print(f"{file_path} not found.")

    return env_dict

