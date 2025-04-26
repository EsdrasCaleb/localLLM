package com.densebrain.rif.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.ByteArrayInputStream;
import org.apache.axis2.util.Base64;

@ExtendWith(MockitoExtension.class)
public class ObjectUtility_serializeObject_0_1_Test {

    @Mock
    private ObjectUtility objectUtility;

    @InjectMocks
    private ObjectUtility objectUtilityUnderTest;

    @Test
    public void testSerializeObject() throws IOException {
        // Arrange
        Object o = new File("test.dat");
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(o);
        }
        // Act
        byte[] result = objectUtilityUnderTest.serializeObject(o);
        // Assert
        assertNotNull(result);
    }

    @Test
    public void testSerializeObject_NullObject() throws IOException {
        // Arrange
        Object o = null;
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(o);
        }
        // Act
        byte[] result = objectUtilityUnderTest.serializeObject(o);
        // Assert
        assertNull(result);
    }

    @Test
    public void testSerializeObject_EmptyObject() throws IOException {
        // Arrange
        Object o = new Object();
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(o);
        }
        // Act
        byte[] result = objectUtilityUnderTest.serializeObject(o);
        // Assert
        assertNotNull(result);
    }

    @Test
    public void testSerializeObject_ObjectWithFields() throws IOException {
        // Arrange
        Object o = new File("test.dat");
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(o);
        }
        // Act
        byte[] result = objectUtilityUnderTest.serializeObject(o);
        // Assert
        assertNotNull(result);
    }

    @Test
    public void testSerializeObject_ObjectWithPrivateFields() throws IOException {
        // Arrange
        Object o = new File("test.dat");
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(o);
        }
        // Act
        byte[] result = objectUtilityUnderTest.serializeObject(o);
        // Assert
        assertNotNull(result);
    }

    @Test
    public void testSerializeObject_ObjectWithPrivateMethods() throws IOException {
        // Arrange
        Object o = new File("test.dat");
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(o);
        }
        // Act
        byte[] result = objectUtilityUnderTest.serializeObject(o);
        // Assert
        assertNotNull(result);
    }
}
