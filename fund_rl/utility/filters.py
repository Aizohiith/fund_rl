import numpy as np

def Fast_EMA(Old_Value : float, New_Value : float, Sensitivity : float = 1.0, Max_Alpha : float = 1.0, Min_Alpha : float = 0.2):
    """
    Computes a dynamically adjusted Exponential Moving Average (EMA) based on the sensitivity to changes in the input values.
    This function adjusts the smoothing factor (Alpha) based on the magnitude of change between the old and new values.
    Args:
        Old_Value (float): The previous EMA value.
        New_Value (float): The new data point to incorporate into the EMA.
        Sensitivity (float): A parameter that controls how sensitive the EMA is to changes in the input values. Higher values make the EMA more responsive.
        Max_Alpha (float): The maximum allowable value for the smoothing factor Alpha.
        Min_Alpha (float): The minimum allowable value for the smoothing factor Alpha.
    Returns:
        float: The updated EMA value.
    """
    Error = abs(New_Value - Old_Value)
    Dynamic_Alpha = 1 - (Error / (Sensitivity + Error + 1e-8))
    Dynamic_Alpha = np.clip(Dynamic_Alpha, Min_Alpha, Max_Alpha)

    return float(Old_Value + Dynamic_Alpha * (New_Value - Old_Value))

def EMA(Old_Value : float, New_Value : float, Alpha : float = 0.99):
    """
    Computes the Exponential Moving Average (EMA) of a new value given an old value and a smoothing factor.
    Args:
        Old_Value (float): The previous EMA value.
        New_Value (float): The new data point to incorporate into the EMA.
        Alpha (float): The smoothing factor for the EMA.
    Returns:
        float: The updated EMA value.
    """
    return float((Old_Value * Alpha)) + (New_Value * ( 1 - Alpha))

def EMA_Filter(Data : list, Alpha : float = 0.99):
    """
    Applies an Exponential Moving Average (EMA) filter to a list of numerical data.
    Args:
        Data (list): A list of numerical values to be filtered.
        Alpha (float): The smoothing factor for the EMA.
    Returns:
        list: A list containing the EMA-filtered values.
    """
    larr_Result = []
    lf_Value = 0
    for C1 in Data:
        lf_Value = EMA(lf_Value, C1, Alpha)
        larr_Result.append(lf_Value)
    return larr_Result

def Parrallel_Average_Filter(Data : list):
    """
    Computes the element-wise average of a list of lists (2D array).
    Args:
        Data (list): A list of lists where each inner list contains numerical values.
    Returns:
        list: A list containing the element-wise average of the input lists.
    """
    Result = []
    Value = 0
    for C1 in range(len(Data[0])):
        Value = 0
        for C2 in Data:
            Value += C2[C1]
        Value /= len(Data)
        Result.append(Value)
    return Result

def Mean_Filter(Data : list):
    """
    Computes the cumulative mean of a list of numerical data.
    Args:
        Data (list): A list of numerical values.
    Returns:
        list: A list containing the cumulative mean values.
    """
    larr_Result = []
    lf_Sum = 0
    for C1 in range(len(Data)):
        lf_Sum += Data[C1]
        larr_Result.append(lf_Sum / (C1 + 1))
    return larr_Result

def STD_Filter(Data : list):
    """
    Computes the cumulative standard deviation of a list of numerical data.
    Args:
        Data (list): A list of numerical values.
    Returns:
        list: A list containing the cumulative standard deviation values.
    """
    larr_Result = []
    larr_Values = []
    for C1 in range(len(Data)):
        larr_Values.append(Data[C1])
        larr_Result.append(float(np.std(larr_Values)))
    return larr_Result