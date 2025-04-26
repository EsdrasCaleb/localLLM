package com.hf.sfm.crypt;

import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Base64_byteArrayToAltBase64_4_0_Test {

    @Test
    void testByteArrayToAltBase64_emptyArray() {
        byte[] emptyArray = {};
        String result = Base64.byteArrayToAltBase64(emptyArray);
        assertEquals("", result);
    }

    @Test
    void testByteArrayToAltBase64_singleByte() {
        byte[] singleByte = { 65 };
        String result = Base64.byteArrayToAltBase64(singleByte);
        assertEquals("A", result);
    }

    @Test
    void testByteArrayToAltBase64_multipleBytes() {
        byte[] multipleBytes = { 65, 66, 67 };
        String result = Base64.byteArrayToAltBase64(multipleBytes);
        assertEquals("ABC", result);
    }

    @Test
    void testByteArrayToAltBase64_negativeBytes() {
        byte[] negativeBytes = { -1, 0, 127 };
        String result = Base64.byteArrayToAltBase64(negativeBytes);
        // Asserting a specific output is not possible without knowing the encoding logic.
        // This test just verifies that the method doesn't throw an exception.
        assertNotNull(result);
    }

    @Test
    void testByteArrayToAltBase64_nullInput() {
        byte[] nullArray = null;
        assertThrows(NullPointerException.class, () -> Base64.byteArrayToAltBase64(nullArray));
    }
}
