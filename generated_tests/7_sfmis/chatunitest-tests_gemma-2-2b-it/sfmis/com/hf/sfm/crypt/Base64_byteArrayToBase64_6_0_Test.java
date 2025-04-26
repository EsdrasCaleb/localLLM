package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_byteArrayToBase64_6_0_Test {

    @Test
    void byteArrayToBase64_Test() {
        // Arrange
        byte[] input = { 1, 2, 3, 4, 5, 6 };
        // Act
        String output = Base64.byteArrayToBase64(input);
        // Assert
        assertEquals("SGVsbG8sIHdvcmxkIQ==", output);
    }
}
