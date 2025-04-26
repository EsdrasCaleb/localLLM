package com.densebrain.rif.util;

import java.io.*;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.axis2.util.Base64;

public class ObjectUtility_deserializeObject_3_0_Test {

    @Test
    void testDeserializeObjectSuccess() throws IOException, ClassNotFoundException {
        // Create a sample object
        TestClass testObject = new TestClass("Test");
        // Serialize the object
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream(baos);
        oos.writeObject(testObject);
        oos.close();
        byte[] bytes = baos.toByteArray();
        // Deserialize the object using the focal method
        Object deserializedObject = ObjectUtility.deserializeObject(bytes);
        // Assert that the deserialized object is equal to the original object
        assertEquals(testObject, deserializedObject);
    }

    @Test
    void testDeserializeObjectClassNotFoundException() throws IOException {
        // Create a byte array that will cause a ClassNotFoundException
        byte[] bytes = { 1, 2, 3, 4, 5 };
        // Assert that a IOException is thrown
        assertThrows(IOException.class, () -> ObjectUtility.deserializeObject(bytes));
    }

    @Test
    void testDeserializeObjectNullBytes() {
        assertThrows(NullPointerException.class, () -> ObjectUtility.deserializeObject(null));
    }

    @Test
    void testDeserializeObjectEmptyBytes() throws IOException {
        byte[] emptyBytes = new byte[0];
        assertThrows(IOException.class, () -> ObjectUtility.deserializeObject(emptyBytes));
    }

    static class TestClass implements Serializable {

        String value;

        TestClass(String value) {
            this.value = value;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null || getClass() != obj.getClass())
                return false;
            TestClass testClass = (TestClass) obj;
            return value.equals(testClass.value);
        }

        @Override
        public int hashCode() {
            return value.hashCode();
        }
    }
}
