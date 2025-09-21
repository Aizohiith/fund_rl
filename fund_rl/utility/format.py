def Format(Strings: list) -> str:
    """
    Format a list of strings into a bordered block for better readability.
    Args:
        Strings (list): List of strings to format.
    Returns:
        str: A single formatted string with borders.
    """
    # Check if the input is empty
    if not Strings:
        return ""

    # Calculate the length of the longest string
    li_Max_length = max(len(s) for s in Strings)

    # Format each string with padding and borders
    larr_Formated_Strings = [f"# {s.ljust(li_Max_length)} #" for s in Strings]

    # Create the top and bottom border
    ls_Boarder = "#" * (li_Max_length + 4)

    # Combine everything into a single string
    ls_Result = f"{ls_Boarder}\n" + "\n".join(larr_Formated_Strings) + f"\n{ls_Boarder}"
    return ls_Result