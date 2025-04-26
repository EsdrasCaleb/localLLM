package com.densebrain.rif.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import org.apache.axis2.util.Base64;

public class ObjectUtility_decodeString_4_0_Test {

    @Test
    public void testDecodeString() {
        // Arrange
        String input = "SGVsbG8sIFdvcmxkIQ==";
        byte[] expectedOutput = "Hello world".getBytes();
        // Act
        byte[] actualOutput = ObjectUtility.decodeString(input);
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testDecodeStringWithInvalidInput() {
        // Arrange
        String input = "invalid_input";
        byte[] expectedOutput = null;
        // Act
        byte[] actualOutput = ObjectUtility.decodeString(input);
        // Assert
        assertEquals(expectedOutput, actualOutput);
    }
}
