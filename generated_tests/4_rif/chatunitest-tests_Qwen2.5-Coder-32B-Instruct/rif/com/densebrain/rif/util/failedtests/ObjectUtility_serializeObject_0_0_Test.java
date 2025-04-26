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

    @Test
    public void testSerializeObject_ObjectOutputStreamCloseThrowsException() throws IOException {
        // Arrange
        String testString = "Test String";
        ByteArrayOutputStream mockBaos = Mockito.mock(ByteArrayOutputStream.class);
        ObjectOutputStream mockOos = Mockito.mock(ObjectOutputStream.class);
        try (MockedConstruction<ObjectOutputStream> mocked = Mockito.mockConstruction(ObjectOutputStream.class, (mock, context) -> {
            assertEquals(mockBaos, context.arguments().get(0));
            doNothing().when(mock).writeObject(testString);
        })) {
            doThrow(new IOException("Mocked IOException")).when(mockOos).close();
            doNothing().when(mockBaos).close();
            when(mockBaos.toByteArray()).thenReturn(new byte[] { 1, 2, 3 });
            // Act & Assert
            assertThrows(IOException.class, () -> ObjectUtility.serializeObject(testString));
            assertTrue(mocked.constructed().contains(mockOos));
            verify(mockOos).writeObject(testString);
            verify(mockOos).close();
            verify(mockBaos).close();
        }
    }

    @Test
    public void testSerializeObject_ByteArrayOutputStreamCloseThrowsException() throws IOException {
        // Arrange
        String testString = "Test String";
        ByteArrayOutputStream mockBaos = Mockito.mock(ByteArrayOutputStream.class);
        ObjectOutputStream mockOos = Mockito.mock(ObjectOutputStream.class);
        try (MockedConstruction<ObjectOutputStream> mocked = Mockito.mockConstruction(ObjectOutputStream.class, (mock, context) -> {
            assertEquals(mockBaos, context.arguments().get(0));
            doNothing().when(mock).writeObject(testString);
        })) {
            doNothing().when(mockOos).close();
            doThrow(new IOException("Mocked IOException")).when(mockBaos).close();
            when(mockBaos.toByteArray()).thenReturn(new byte[] { 1, 2, 3 });
            // Act & Assert
            assertThrows(IOException.class, () -> ObjectUtility.serializeObject(testString));
            assertTrue(mocked.constructed().contains(mockOos));
            verify(mockOos).writeObject(testString);
            verify(mockOos).close();
            verify(mockBaos).close();
        }
    }
}
