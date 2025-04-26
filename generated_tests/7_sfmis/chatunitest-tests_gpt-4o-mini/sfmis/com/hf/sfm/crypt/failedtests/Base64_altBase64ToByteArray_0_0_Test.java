package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_altBase64ToByteArray_0_0_Test {

    @Test
    public void testAltBase64ToByteArray_ValidInput() {
        // Given
        // Base64 for "Hello World"
        String input = "SGVsbG8gV29ybGQ=";
        byte[] expectedOutput = "Hello World".getBytes();
        // When
        byte[] actualOutput = Base64.altBase64ToByteArray(input);
        // Then
        assertArrayEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testAltBase64ToByteArray_EmptyInput() {
        // Given
        String input = "";
        byte[] expectedOutput = new byte[0];
        // When
        byte[] actualOutput = Base64.altBase64ToByteArray(input);
        // Then
        assertArrayEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testAltBase64ToByteArray_InvalidInput() {
        // Given
        String input = "InvalidBase64@!";
        // When
        byte[] actualOutput = Base64.altBase64ToByteArray(input);
        // Then
        // Assuming the method handles invalid input gracefully, you can assert an empty array or handle as needed
        assertArrayEquals(new byte[0], actualOutput);
    }

    @Test
    public void testAltBase64ToByteArray_NullInput() {
        // Given
        String input = null;
        // When
        byte[] actualOutput = Base64.altBase64ToByteArray(input);
        // Then
        // Assuming the method handles null input gracefully, you can assert an empty array or handle as needed
        assertArrayEquals(new byte[0], actualOutput);
    }
}
