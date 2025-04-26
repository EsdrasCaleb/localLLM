package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_base64ToByteArray_2_0_Test {

    @Test
    public void testBase64ToByteArray() {
        String base64String = "SGVsbG8gd29ybGQ=";
        byte[] expectedBytes = { 72, 101, 108, 108, 111, 32, 116, 104, 101, 114 };
        byte[] actualBytes = Base64.base64ToByteArray(base64String);
        assertEquals(expectedBytes.length, actualBytes.length);
        for (int i = 0; i < expectedBytes.length; i++) {
            assertEquals(expectedBytes[i], actualBytes[i]);
        }
    }
}
