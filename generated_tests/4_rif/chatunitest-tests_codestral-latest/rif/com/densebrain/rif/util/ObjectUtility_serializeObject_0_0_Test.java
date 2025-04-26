package com.densebrain.rif.util;

import java.io.IOException;
import java.io.NotSerializableException;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import org.apache.axis2.util.Base64;

class ObjectUtility_serializeObject_0_0_Test {

    @Test
    void testSerializeObject() throws IOException {
        // Test with a serializable object
        SerializableObject obj = new SerializableObject();
        byte[] serializedData = ObjectUtility.serializeObject(obj);
        assertNotNull(serializedData);
        assertTrue(serializedData.length > 0);
        // Test with a non-serializable object
        NonSerializableObject nonSerializableObj = new NonSerializableObject();
        assertThrows(NotSerializableException.class, () -> {
            ObjectUtility.serializeObject(nonSerializableObj);
        });
        // Test with null object
        assertThrows(NullPointerException.class, () -> {
            ObjectUtility.serializeObject(null);
        });
    }

    // Serializable test object
    private static class SerializableObject implements Serializable {

        private static final long serialVersionUID = 1L;

        private String data = "test data";
    }

    // Non-serializable test object
    private static class NonSerializableObject {

        private String data = "test data";
    }
}
