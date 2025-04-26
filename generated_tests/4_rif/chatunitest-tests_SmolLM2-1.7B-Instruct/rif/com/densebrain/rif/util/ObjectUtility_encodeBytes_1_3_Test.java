package com.densebrain.rif.util;

// <Buggy Line>: a type with the same simple name is already defined by the single-type-import of java.util.Base64
import org.apache.axis2.util.Base64;
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

@ExtendWith(MockitoExtension.class)
class ObjectUtility_encodeBytes_1_3_Test {

    @InjectMocks
    private ObjectUtility objectUtility;

    @Mock
    private Base64 base64;

    @Test
    void testEncodeBytes() {
        // Arrange
        byte[] bytes = new byte[] { 'a', 'b', 'c' };
        // Act
        String result = objectUtility.encodeBytes(bytes);
        // Assert
        assertEquals("VGhpcyBpcyBhIHRlc3Qgc3RyaW5nLg==", result);
    }
}
