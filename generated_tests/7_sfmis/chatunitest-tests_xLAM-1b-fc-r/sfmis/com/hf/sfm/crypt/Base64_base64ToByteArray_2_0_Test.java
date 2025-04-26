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
        String input = "SGVsbG8gd29ybGQ=";
        byte[] expected = { 72, 101, 32, 116, 101, 115, 116, 32, 97, 114, 116, 105, 111, 110, 33 };
        byte[] actual = Base64.base64ToByteArray(input);
        assertArrayEquals(expected, actual);
        input = "ZW5jb2RlZCB0ZXh0";
        expected = new byte[] { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
        actual = Base64.base64ToByteArray(input);
        assertArrayEquals(expected, actual);
        input = "ZW5jb2RlZCB0ZXh0ZXh0ZXh0";
        expected = new byte[] { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
        actual = Base64.base64ToByteArray(input);
        assertArrayEquals(expected, actual);
        input = "ZW5jb2RlZCB0ZXh0ZXh0ZXh0ZXh0ZXh0";
        expected = new byte[] { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
        actual = Base64.base64ToByteArray(input);
        assertArrayEquals(expected, actual);
        input = "ZW5jb2RlZCB0ZXh0ZXh0ZXh0ZXh0ZXh0ZXh0ZXh0";
        expected = new byte[] { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 };
        actual = Base64.base64ToByteArray(input);
        assertArrayEquals(expected, actual);
    }
}
