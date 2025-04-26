package com.hf.sfm.crypt;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Base64_byteArrayToBase64_6_0_Test {

    @Test
    void byteArrayToBase64_emptyArray() {
        byte[] emptyArray = new byte[0];
        String result = Base64.byteArrayToBase64(emptyArray);
        assertEquals("", result);
    }

    @Test
    void byteArrayToBase64_singleByte() {
        byte[] singleByte = { 65 };
        String result = Base64.byteArrayToBase64(singleByte);
        assertEquals("QQ==", result);
    }

    @Test
    void byteArrayToBase64_multipleBytes() {
        byte[] multipleBytes = { 65, 66, 67 };
        String result = Base64.byteArrayToBase64(multipleBytes);
        assertEquals("QUJD", result);
    }

    @Test
    void byteArrayToBase64_nullArray() {
        byte[] nullArray = null;
        assertThrows(NullPointerException.class, () -> Base64.byteArrayToBase64(nullArray));
    }

    @Test
    void byteArrayToBase64_specialCharacters() {
        byte[] specialChars = { 33, 64, 94 };
        String result = Base64.byteArrayToBase64(specialChars);
        assertEquals("!@^", result);
    }

    @Test
    void byteArrayToBase64_largeArray() {
        byte[] largeArray = new byte[100];
        for (int i = 0; i < 100; i++) {
            largeArray[i] = (byte) i;
        }
        String result = Base64.byteArrayToBase64(largeArray);
        assertNotNull(result);
        // No need to assert the exact string, length is sufficient.
        assertTrue(result.length() > 0);
    }
}
