package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_base64ToByteArray_2_0_Test {

    @Test
    public void testBase64ToByteArray_EmptyString_ReturnsEmptyByteArray() {
        String input = "";
        byte[] expected = new byte[0];
        byte[] actual = Base64.base64ToByteArray(input);
        assertArrayEquals(expected, actual);
    }

    @Test
    public void testBase64ToByteArray_NullString_ThrowsNullPointerException() {
        String input = null;
        assertThrows(NullPointerException.class, () -> Base64.base64ToByteArray(input));
    }

    @Test
    public void testBase64ToByteArray_ValidBase64String_ReturnsCorrectByteArray() {
        String input = "SGVsbG8gd29ybGQh";
        byte[] expected = { 104, 111, 118, 101, 116, 32, 99, 111, 114, 108, 100, 33 };
        byte[] actual = Base64.base64ToByteArray(input);
        assertArrayEquals(expected, actual);
    }

    @Test
    public void testBase64ToByteArray_ValidBase64String_WithMultipleLines_ReturnsCorrectByteArray() {
        String input = "SGVsbG8gd29ybGQh\n" + "aGVsbG8gd29ybGQh";
        byte[] expected = { 104, 111, 118, 101, 116, 32, 99, 111, 114, 108, 100, 33, 97, 116, 104, 101, 108, 100, 33, 97, 116, 104, 101, 108, 100, 33 };
        byte[] actual = Base64.base64ToByteArray(input);
        assertArrayEquals(expected, actual);
    }
}
