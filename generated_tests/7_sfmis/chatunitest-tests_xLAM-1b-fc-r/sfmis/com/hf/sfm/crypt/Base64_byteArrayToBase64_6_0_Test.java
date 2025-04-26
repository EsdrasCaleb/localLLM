package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_byteArrayToBase64_6_0_Test {

    @Test
    public void testByteArrayToBase64() {
        byte[] input = { 1, 2, 3, 4, 5 };
        String expectedOutput = "ZW5jb2RlZCB0ZXh0";
        assertEquals(expectedOutput, Base64.byteArrayToBase64(input));
    }
}
