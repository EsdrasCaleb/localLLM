package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_altBase64ToByteArray_0_3_Test {

    @Test
    public void testAltBase64ToByteArray_SimpleString() {
        String input = "Hello, World!";
        byte[] expectedOutput = Base64.altBase64ToByteArray(input);
        byte[] actualOutput = Base64.altBase64ToByteArray(input);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testAltBase64ToByteArray_SingleCharacter() {
        char input = 'A';
        byte[] expectedOutput = Base64.altBase64ToByteArray(String.valueOf(input));
        byte[] actualOutput = Base64.altBase64ToByteArray(String.valueOf(input));
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testAltBase64ToByteArray_EmptyString() {
        String input = "";
        byte[] expectedOutput = Base64.altBase64ToByteArray(input);
        byte[] actualOutput = Base64.altBase64ToByteArray(input);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testAltBase64ToByteArray_NullString() {
        String input = null;
        try {
            Base64.altBase64ToByteArray(input);
            fail("Expected NullPointerException");
        } catch (NullPointerException e) {
            // expected
        }
    }

    @Test
    public void testAltBase64ToByteArray_EmptyInput() {
        String input = "";
        byte[] expectedOutput = Base64.altBase64ToByteArray(input);
        byte[] actualOutput = Base64.altBase64ToByteArray(input);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testAltBase64ToByteArray_InvalidInput() {
        String input = "InvalidBase64String";
        byte[] expectedOutput = Base64.altBase64ToByteArray(input);
        byte[] actualOutput = Base64.altBase64ToByteArray(input);
        assertEquals(expectedOutput, actualOutput);
    }
}
