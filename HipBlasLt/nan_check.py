import numpy as np

def check_for_nan_infinite(array1, array2):
    """
    Check if any of the three NumPy arrays contain NaN or infinite values.
    Prints which array(s) contain NaN or infinite values.

    Parameters:
    array1 (numpy.ndarray): First array to check.
    array2 (numpy.ndarray): Second array to check.
    array3 (numpy.ndarray): Third array to check.

    Returns:
    bool: True if any array contains NaN or infinite values, False otherwise.
    """
    # Check each array for non-finite values (NaN or inf)
    is_finite1 = (array1 & 0x7F80 == 0x7F80) & (array1 & 0x007F != 0) #np.isfinite(array1).all()
    is_finite2 = (array2 & 0x7F80 == 0x7F80) & (array2 & 0x007F != 0) #np.isfinite(array2).all()
    # is_finite3 = np.isfinite(array3).all()

    if np.any(is_finite1):
        print("array1 has nan!")
    elif np.any(is_finite2):
        print("array2 has nan!")
    else:
        print("NO NAN!")
# Example usage
array1 = np.fromfile('input_bin_file/ok_bin/matrix_A.bin', dtype='uint16')  # Load array from binary file
array2 = np.fromfile('input_bin_file/ok_bin/matrix_B.bin', dtype='uint16')  # Load array from binary file
# array3 = np.fromfile('matrix_C.bin', dtype='uint16')  # Load array from binary file

# Check for NaN or infinite values
check_for_nan_infinite(array1, array2)