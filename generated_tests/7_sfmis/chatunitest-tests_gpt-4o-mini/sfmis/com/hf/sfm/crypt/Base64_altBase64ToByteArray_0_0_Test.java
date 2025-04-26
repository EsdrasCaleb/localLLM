package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_altBase64ToByteArray_0_0_Test {

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
}
