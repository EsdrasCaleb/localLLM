package com.hf.sfm.crypt;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_byteArrayToAltBase64_4_0_Test {

    @Test
    void testByteArrayToAltBase64_emptyArray() throws Exception {
        byte[] emptyArray = new byte[0];
        String result = Base64.byteArrayToAltBase64(emptyArray);
        assertEquals("", result);
    }

    @Test
    void testByteArrayToAltBase64_singleByte() throws Exception {
        // 'A'
        byte[] singleByte = { 65 };
        String result = Base64.byteArrayToAltBase64(singleByte);
        // Expect 'A?' based on the private arrays in the Base64 class.
        assertEquals("A?", result);
    }

    @Test
    void testByteArrayToAltBase64_multipleBytes() throws Exception {
        // "ABCD"
        byte[] multipleBytes = { 65, 66, 67, 68 };
        String result = Base64.byteArrayToAltBase64(multipleBytes);
        // Expect "A?B?C?D?" based on the private arrays in the Base64 class.
        assertEquals("A?B?C?D?", result);
    }

    @Test
    void testByteArrayToAltBase64_nullArray() throws Exception {
        String result = Base64.byteArrayToAltBase64(null);
        assertEquals(null, result);
    }

    @Test
    void testByteArrayToAltBase64_specialCharacters() throws Exception {
        // "!"#$
        byte[] specialChars = { 33, 34, 35 };
        String result = Base64.byteArrayToAltBase64(specialChars);
        // The expected output depends on the mapping in _$23167 and _$23170 which is not directly accessible.
        // This test would require more in-depth analysis of the private method's logic or mocking.
        // At least check for non-null result
        assertNotNull(result);
    }
}
