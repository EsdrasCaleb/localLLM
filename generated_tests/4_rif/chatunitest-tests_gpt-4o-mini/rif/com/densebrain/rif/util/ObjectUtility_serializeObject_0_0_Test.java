package com.densebrain.rif.util;

import java.io.IOException;
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

public class ObjectUtility_serializeObject_0_0_Test {

    @Test
    public void testSerializeObject_withSerializableObject() throws IOException {
        // Arrange
        TestObject testObject = new TestObject("Test", 123);
        // Act
        byte[] result = ObjectUtility.serializeObject(testObject);
        // Assert
        assertArrayEquals(testObject.getSerializedForm(), result);
    }

    @Test
    public void testSerializeObject_withNullObject() {
        // Act & Assert
        assertThrows(IOException.class, () -> {
            ObjectUtility.serializeObject(null);
        });
    }

    @Test
    public void testSerializeObject_withNonSerializableObject() {
        // Arrange
        NonSerializableObject nonSerializableObject = new NonSerializableObject();
        // Act & Assert
        assertThrows(IOException.class, () -> {
            ObjectUtility.serializeObject(nonSerializableObject);
        });
    }

    private static class TestObject implements Serializable {

        private String name;

        private int value;

        public TestObject(String name, int value) {
            this.name = name;
            this.value = value;
        }

        public byte[] getSerializedForm() throws IOException {
            return ObjectUtility.serializeObject(this);
        }
    }

    private static class NonSerializableObject {

        private String data = "Non-serializable data";
    }
}
