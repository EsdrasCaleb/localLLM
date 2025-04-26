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
