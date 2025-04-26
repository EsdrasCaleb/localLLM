package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_base64ToByteArray_2_2_Test {

    @Test
    public void testBase64ToByteArray() {
        String base64String = "ZmlsZSBzdHJpbmc=";
        byte[] expected = { 65, 66, 67, 68 };
        byte[] actual = Base64.base64ToByteArray(base64String);
        assertArrayEquals(expected, actual);
    }
}
