package com.hf.sfm.crypt;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Base64_byteArrayToBase64_6_2_Test {

    @Mock
    private byte[] inputByteArray;

    @Test
    public void testByteArrayToBase64_WithValidByteArray_ReturnsValidBase64String() {
        // Act
        String result = Base64.byteArrayToBase64(inputByteArray);
        // Assert
        assertEquals("Valid Base64 String", result);
    }

    @Test
    public void testByteArrayToBase64_WithNullByteArray_ThrowsNullPointerException() {
        // Act and Assert
        assertThrows(NullPointerException.class, () -> Base64.byteArrayToBase64(null));
    }

    @Test
    public void testByteArrayToBase64_WithEmptyByteArray_ReturnsEmptyBase64String() {
        // Act
        String result = Base64.byteArrayToBase64(new byte[0]);
        // Assert
        assertEquals("", result);
    }

    @Test
    public void testByteArrayToBase64_WithByteArrayContainingNonBase64Characters_ThrowsIllegalArgumentException() {
        // Act and Assert
        assertThrows(IllegalArgumentException.class, () -> Base64.byteArrayToBase64(new byte[] { 1, 2, 3, 4, 5 }));
    }
}
