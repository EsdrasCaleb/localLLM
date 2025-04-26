package com.hf.sfm.crypt;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Base64_main_7_1_Test {

    @InjectMocks
    private Base64 focal;

    @Test
    public void testBase64EncodingDecoding() {
        // Arrange
        String input = "0123456789";
        byte[] inputBytes = input.getBytes();
        String expectedEncoded = "SGVsbG8gd29ybGQh";
        String expectedOriginal = input;
        // Act
        String encoded = focal.byteArrayToBase64(inputBytes);
        byte[] decodedBytes = focal.base64ToByteArray(encoded);
        // Assert
        assertEquals(expectedEncoded, encoded);
        assertEquals(expectedOriginal, new String(decodedBytes));
    }

    @Test
    public void testBase64EncodingDecodingWithNonBase64String() {
        // Arrange
        String input = "Hello World!";
        byte[] inputBytes = input.getBytes();
        String expectedEncoded = "SGVsbG8gd29ybGQh";
        String expectedOriginal = input;
        // Act
        String encoded = focal.byteArrayToBase64(inputBytes);
        byte[] decodedBytes = focal.base64ToByteArray(encoded);
        // Assert
        assertEquals(expectedEncoded, encoded);
        assertEquals(expectedOriginal, new String(decodedBytes));
    }

    @Test
    public void testBase64EncodingDecodingWithEmptyString() {
        // Arrange
        String input = "";
        byte[] inputBytes = input.getBytes();
        String expectedEncoded = "";
        String expectedOriginal = input;
        // Act
        String encoded = focal.byteArrayToBase64(inputBytes);
        byte[] decodedBytes = focal.base64ToByteArray(encoded);
        // Assert
        assertEquals(expectedEncoded, encoded);
        assertEquals(expectedOriginal, new String(decodedBytes));
    }

    @Test
    public void testBase64EncodingDecodingWithNullInput() {
        // Act and Assert
        assertThrows(NullPointerException.class, () -> focal.byteArrayToBase64(null));
        assertThrows(NullPointerException.class, () -> focal.base64ToByteArray(null));
    }
}
