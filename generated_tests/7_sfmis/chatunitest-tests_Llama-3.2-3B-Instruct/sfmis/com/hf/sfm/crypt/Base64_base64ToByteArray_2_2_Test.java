package com.hf.sfm.crypt;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Base64_base64ToByteArray_2_2_Test {

    @Mock
    private String encodedString;

    @InjectMocks
    private Base64 base64;

    @Test
    public void testBase64ToByteArray() {
        // Arrange
        // Base64 encoded string
        when(encodedString.getBytes()).thenReturn("SGVsbG8gd29ybGQh".getBytes());
        // Act
        byte[] result = base64.base64ToByteArray(encodedString);
        // Assert
        assertArrayEquals("Hello World".getBytes(), result);
    }

    @Test
    public void testBase64ToByteArray_InvalidInput() {
        // Arrange
        when(encodedString.getBytes()).thenReturn(null);
        // Act and Assert
        assertThrows(NullPointerException.class, () -> base64.base64ToByteArray(encodedString));
    }

    @Test
    public void testBase64ToByteArray_EmptyString() {
        // Arrange
        when(encodedString.getBytes()).thenReturn(new byte[0]);
        // Act
        byte[] result = base64.base64ToByteArray(encodedString);
        // Assert
        assertArrayEquals(new byte[0], result);
    }
}
