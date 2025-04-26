package com.hf.sfm.crypt;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Base64_byteArrayToBase64_6_0_Test {

    @Test
    public void testByteArrayToBase64_EmptyArray_ReturnsEmptyString() {
        byte[] input = {};
        String expected = "";
        String actual = Base64.byteArrayToBase64(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testByteArrayToBase64_NullArray_ThrowsNullPointerException() {
        byte[] input = null;
        assertThrows(NullPointerException.class, () -> Base64.byteArrayToBase64(input));
    }

    @Test
    public void testByteArrayToBase64_SingleByteArray_ReturnsCorrectBase64String() {
        byte[] input = { 1 };
        String expected = "A";
        String actual = Base64.byteArrayToBase64(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testByteArrayToBase64_MultipleByteArray_ReturnsCorrectBase64String() {
        byte[] input = { 1, 2, 3 };
        String expected = "AA";
        String actual = Base64.byteArrayToBase64(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testByteArrayToBase64_LargeByteArray_ReturnsCorrectBase64String() {
        byte[] input = new byte[10];
        for (int i = 0; i < 10; i++) {
            input[i] = (byte) i;
        }
        String expected = "AAAAAA";
        String actual = Base64.byteArrayToBase64(input);
        assertEquals(expected, actual);
    }
}
