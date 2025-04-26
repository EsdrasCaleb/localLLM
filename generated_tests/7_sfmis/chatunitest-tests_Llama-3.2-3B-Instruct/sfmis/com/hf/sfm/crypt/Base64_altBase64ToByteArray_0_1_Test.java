package com.hf.sfm.crypt;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Base64_altBase64ToByteArray_0_1_Test {

    @InjectMocks
    private Base64 base64;

    @Test
    public void testAltBase64ToByteArray_EmptyString() {
        // Arrange
        String input = "";
        byte[] expected = new byte[0];
        // Act
        byte[] actual = base64.altBase64ToByteArray(input);
        // Assert
        assertArrayEquals(expected, actual);
    }

    @Test
    public void testAltBase64ToByteArray_NullInput() {
        // Act and Assert
        assertThrows(NullPointerException.class, () -> base64.altBase64ToByteArray(null));
    }
}
