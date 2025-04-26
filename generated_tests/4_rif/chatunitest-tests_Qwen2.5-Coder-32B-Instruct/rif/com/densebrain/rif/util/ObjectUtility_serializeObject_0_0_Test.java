package com.densebrain.rif.util;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayInputStream;
import java.io.ObjectInputStream;
import org.apache.axis2.util.Base64;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ObjectUtility_serializeObject_0_0_Test {

    @Test
    public void testSerializeObject_Success() throws IOException {
        // Arrange
        String testString = "Test String";
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(testString);
        }
        byte[] expectedBytes = baos.toByteArray();
        // Act
        byte[] result = ObjectUtility.serializeObject(testString);
        // Assert
        assertNotNull(result);
        assertArrayEquals(expectedBytes, result);
    }
}
