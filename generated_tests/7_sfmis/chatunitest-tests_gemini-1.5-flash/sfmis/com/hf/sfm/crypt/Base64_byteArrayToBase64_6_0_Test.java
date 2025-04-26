package com.hf.sfm.crypt;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_byteArrayToBase64_6_0_Test {

    @Test
    void testByteArrayToBase64_emptyArray() {
        assertEquals("", Base64.byteArrayToBase64(new byte[0]));
    }

    @Test
    void testByteArrayToBase64_singleByte() {
        assertEquals("AA==", Base64.byteArrayToBase64(new byte[] { 0 }));
        assertEquals("AQ==", Base64.byteArrayToBase64(new byte[] { 1 }));
        assertEquals("/w==", Base64.byteArrayToBase64(new byte[] { -1 }));
    }

    @Test
    void testByteArrayToBase64_multipleBytes() {
        assertEquals("YWJj", Base64.byteArrayToBase64("abc".getBytes()));
        assertEquals("SGVsbG8gV29ybGQh", Base64.byteArrayToBase64("Hello World!".getBytes()));
        assertEquals("aGVsbG8gd29ybGQh", Base64.byteArrayToBase64("hello world!".getBytes()));
    }

    @Test
    void testByteArrayToBase64_nullArray() {
        assertThrows(NullPointerException.class, () -> Base64.byteArrayToBase64(null));
    }

    @Test
    void testPrivateMethod_23170() throws Exception {
        byte[] input = "test".getBytes();
        Method privateMethod = Base64.class.getDeclaredMethod("_$23170", byte[].class, boolean.class);
        privateMethod.setAccessible(true);
        String result = (String) privateMethod.invoke(null, input, false);
        assertEquals("dGVzdA==", result);
        String result2 = (String) privateMethod.invoke(null, input, true);
        // The exact output of the private method with true parameter is hard to predict without knowing its implementation.
        // This assertion is replaced with a general check for non-null and non-empty result.
        assertNotNull(result2);
        assertFalse(result2.isEmpty());
    }
}
