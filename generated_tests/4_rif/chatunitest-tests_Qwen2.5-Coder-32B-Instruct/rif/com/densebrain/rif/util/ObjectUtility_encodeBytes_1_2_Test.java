package com.densebrain.rif.util;

import java.lang.reflect.Method;
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

public class ObjectUtility_encodeBytes_1_2_Test {

    @Test
    public void testEncodeBytes() throws Exception {
        // Arrange
        byte[] inputBytes = "Hello, World!".getBytes();
        String expectedEncodedString = "SGVsbG8sIFdvcmxkIQ==";
        // Use reflection to invoke the encodeBytes method
        Method encodeBytesMethod = ObjectUtility.class.getDeclaredMethod("encodeBytes", byte[].class);
        encodeBytesMethod.setAccessible(true);
        // Act
        String actualEncodedString = (String) encodeBytesMethod.invoke(null, inputBytes);
        // Assert
        assertEquals(expectedEncodedString, actualEncodedString, "The Base64 encoded string should match the expected value");
    }
}
