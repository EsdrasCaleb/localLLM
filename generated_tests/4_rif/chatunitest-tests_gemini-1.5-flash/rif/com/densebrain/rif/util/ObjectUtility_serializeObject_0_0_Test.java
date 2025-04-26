package com.densebrain.rif.util;

import java.io.*;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.apache.axis2.util.Base64;

public class ObjectUtility_serializeObject_0_0_Test {

    @Test
    void testSerializeObject_simpleObject() throws IOException {
        SimpleObject obj = new SimpleObject("test", 123);
        byte[] serialized = ObjectUtility.serializeObject(obj);
        assertNotNull(serialized);
        assertTrue(serialized.length > 0);
        try (ByteArrayInputStream bais = new ByteArrayInputStream(serialized);
            ObjectInputStream ois = new ObjectInputStream(bais)) {
            SimpleObject deserialized = (SimpleObject) ois.readObject();
            assertEquals(obj.getName(), deserialized.getName());
            assertEquals(obj.getValue(), deserialized.getValue());
        } catch (ClassNotFoundException e) {
            fail("ClassNotFoundException during deserialization: " + e.getMessage());
        }
    }

    @Test
    void testSerializeObject_complexObject() throws IOException, ClassNotFoundException {
        ComplexObject obj = new ComplexObject("test", new SimpleObject("nested", 456));
        byte[] serialized = ObjectUtility.serializeObject(obj);
        assertNotNull(serialized);
        assertTrue(serialized.length > 0);
        try (ByteArrayInputStream bais = new ByteArrayInputStream(serialized);
            ObjectInputStream ois = new ObjectInputStream(bais)) {
            ComplexObject deserialized = (ComplexObject) ois.readObject();
            assertEquals(obj.getName(), deserialized.getName());
            assertEquals(obj.getNested().getName(), deserialized.getNested().getName());
            assertEquals(obj.getNested().getValue(), deserialized.getNested().getValue());
        }
    }

    static class SimpleObject implements Serializable {

        private String name;

        private int value;

        public SimpleObject(String name, int value) {
            this.name = name;
            this.value = value;
        }

        public String getName() {
            return name;
        }

        public int getValue() {
            return value;
        }
    }

    static class ComplexObject implements Serializable {

        private String name;

        private SimpleObject nested;

        public ComplexObject(String name, SimpleObject nested) {
            this.name = name;
            this.nested = nested;
        }

        public String getName() {
            return name;
        }

        public SimpleObject getNested() {
            return nested;
        }
    }

    static class NonSerializableObject {
        // No implementation needed, it's intentionally not serializable
    }
}
