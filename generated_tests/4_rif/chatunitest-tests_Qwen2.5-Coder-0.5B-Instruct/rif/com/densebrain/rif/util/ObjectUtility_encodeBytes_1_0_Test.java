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

public class ObjectUtility_encodeBytes_1_0_Test {

    @Test
    public void testEncodeBytes() {
        // Arrange
        byte[] data = { 0, 1, 2, 3, 4 };
        // Act
        String encodedString = ObjectUtility.encodeBytes(data);
        // Assert
        // Base64 encoded string
        assertEquals("SGVsbG8gV29ybGQ=", encodedString);
    }
}
