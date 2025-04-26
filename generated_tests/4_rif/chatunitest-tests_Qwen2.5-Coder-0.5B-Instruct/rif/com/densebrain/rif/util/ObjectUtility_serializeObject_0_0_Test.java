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

class ObjectUtility_serializeObject_0_0_Test {

    @Test
    public void testSerializeObject() {
        // Given
        Object obj = new Object();
        try {
            // When
            byte[] serializedBytes = ObjectUtility.serializeObject(obj);
            // Then
            assertNotNull(serializedBytes);
            // Assuming the size of an object in bytes is 8
            assertEquals(8, serializedBytes.length);
        } catch (IOException e) {
            fail("Serialization failed: " + e.getMessage());
        }
    }
}
