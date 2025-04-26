package com.hf.sfm.crypt;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Base64_altBase64ToByteArray_0_0_Test {

    @Mock
    private String s;

    @InjectMocks
    private Base64 base64;

    @Test
    public void testAltBase64ToByteArray() {
        // Arrange
        byte[] expected = {/* expected bytes */
        };
        when(s.getBytes()).thenReturn(expected);
        // Act
        byte[] actual = base64.altBase64ToByteArray(s);
        // Assert
        assertArrayEquals(expected, actual);
    }

    @Test
    public void testAltBase64ToByteArray_InvalidInput() {
        // Arrange
        String invalidInput = "Invalid input";
        // Act and Assert
        assertThrows(NullPointerException.class, () -> base64.altBase64ToByteArray(invalidInput));
    }

    @Test
    public void testAltBase64ToByteArray_NullInput() {
        // Arrange
        String nullInput = null;
        // Act and Assert
        assertThrows(NullPointerException.class, () -> base64.altBase64ToByteArray(nullInput));
    }
}
