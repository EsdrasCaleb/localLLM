package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_byteArrayToBase64_6_0_Test {

    @Test
    public void testByteArrayToBase64_EmptyArray() {
        byte[] input = {};
        String expected = "";
        String actual = Base64.byteArrayToBase64(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testByteArrayToBase64_SingleByte() {
        // 'A'
        byte[] input = { 65 };
        String expected = "QQ==";
        String actual = Base64.byteArrayToBase64(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testByteArrayToBase64_TwoBytes() {
        // 'AB'
        byte[] input = { 65, 66 };
        String expected = "QUI=";
        String actual = Base64.byteArrayToBase64(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testByteArrayToBase64_ThreeBytes() {
        // 'ABC'
        byte[] input = { 65, 66, 67 };
        String expected = "QUJD";
        String actual = Base64.byteArrayToBase64(input);
        assertEquals(expected, actual);
    }

    @Test
    public void testByteArrayToBase64_MultipleBytes() {
        // 'Hello'
        byte[] input = { 72, 101, 108, 108, 111 };
        String expected = "SGVsbG8=";
        String actual = Base64.byteArrayToBase64(input);
        assertEquals(expected, actual);
    }
}
