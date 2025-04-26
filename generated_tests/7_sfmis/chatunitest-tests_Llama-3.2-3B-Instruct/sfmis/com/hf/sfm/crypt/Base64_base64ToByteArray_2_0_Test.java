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
}
