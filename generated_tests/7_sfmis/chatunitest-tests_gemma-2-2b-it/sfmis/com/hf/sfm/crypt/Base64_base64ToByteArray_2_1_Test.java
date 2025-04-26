package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_base64ToByteArray_2_1_Test {

    @Test
    void testBase64ToByteArray() {
        Base64 base64 = new Base64();
        String input = "SGVsbG8=";
        byte[] output = base64.base64ToByteArray(input);
        assertArrayEquals(new byte[] { 1 }, output);
    }
}
