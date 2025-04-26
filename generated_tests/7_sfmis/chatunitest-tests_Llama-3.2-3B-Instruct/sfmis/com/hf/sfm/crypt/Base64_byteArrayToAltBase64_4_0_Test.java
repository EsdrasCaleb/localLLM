package com.hf.sfm.crypt;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class Base64_byteArrayToAltBase64_4_0_Test {

    @Mock
    private Base64 base64;

    @InjectMocks
    private Base64 underTest = new Base64();

    @Test
    public void testByteArrayToAltBase64_EmptyArray_ReturnsEmptyString() {
        byte[] input = new byte[0];
        String result = underTest.byteArrayToAltBase64(input);
        assertEquals("", result);
    }

    @Test
    public void testByteArrayToAltBase64_NullArray_ThrowsNullPointerException() {
        assertThrows(NullPointerException.class, () -> underTest.byteArrayToAltBase64(null));
    }

    @Test
    public void testByteArrayToAltBase64_SingleByteArray_ReturnsCorrectBase64String() {
        byte[] input = new byte[] { (byte) 0x12 };
        String result = underTest.byteArrayToAltBase64(input);
        assertEquals("w", result);
    }

    @Test
    public void testByteArrayToAltBase64_MultiByteArray_ReturnsCorrectBase64String() {
        byte[] input = new byte[] { (byte) 0x12, (byte) 0x34, (byte) 0x56, (byte) 0x78 };
        String result = underTest.byteArrayToAltBase64(input);
        assertEquals("wMg==", result);
    }

    @Test
    public void testByteArrayToAltBase64_MultiByteArrayWithSpecialCharacters_ReturnsCorrectBase64String() {
        byte[] input = new byte[] { (byte) 0x12, (byte) 0x34, (byte) 0x56, (byte) 0x78, (byte) 0x1a, (byte) 0x1b, (byte) 0x1c, (byte) 0x1d };
        String result = underTest.byteArrayToAltBase64(input);
        assertEquals("wMg==", result);
    }

    @Test
    public void testByteArrayToAltBase64_InvalidInput_ThrowsException() {
        byte[] input = new byte[] { (byte) 0x12, (byte) 0x34, (byte) 0x56, (byte) 0x78, (byte) 0x1a, (byte) 0x1b, (byte) 0x1c, (byte) 0x1d, (byte) 0x1e, (byte) 0x1f };
        assertThrows(Exception.class, () -> underTest.byteArrayToAltBase64(input));
    }
}
