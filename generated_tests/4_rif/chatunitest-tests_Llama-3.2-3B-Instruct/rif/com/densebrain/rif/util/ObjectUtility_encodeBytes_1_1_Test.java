package com.densebrain.rif.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import org.apache.axis2.util.Base64;

@ExtendWith(MockitoExtension.class)
public class ObjectUtility_encodeBytes_1_1_Test {

    @InjectMocks
    private ObjectUtility objectUtility;

    @Test
    public void testEncodeBytes() {
        // Arrange
        byte[] bytes = "Hello, World!".getBytes();
        // Act
        String encodedBytes = ObjectUtility.encodeBytes(bytes);
        // Assert
        assertEquals("SGVsbG8sIFdvcmxkIQ==", encodedBytes);
    }
}
