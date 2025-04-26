package com.hf.sfm.crypt;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Base64_byteArrayToAltBase64_4_0_Test {

    private static Method _$23170Method;

    @BeforeAll
    static void setUp() throws NoSuchMethodException {
        _$23170Method = Base64.class.getDeclaredMethod("_$23170", byte[].class, boolean.class);
        _$23170Method.setAccessible(true);
    }

    @Test
    void testByteArrayToAltBase64() throws Exception {
        byte[] input = { 1, 2, 3, 4 };
        String expected = (String) _$23170Method.invoke(null, input, true);
        String result = Base64.byteArrayToAltBase64(input);
        assertEquals(expected, result);
    }

    @Test
    void testByteArrayToAltBase64EmptyArray() throws Exception {
        byte[] input = {};
        String expected = (String) _$23170Method.invoke(null, input, true);
        String result = Base64.byteArrayToAltBase64(input);
        assertEquals(expected, result);
    }

    @Test
    void testByteArrayToAltBase64NullInput() {
        assertThrows(NullPointerException.class, () -> Base64.byteArrayToAltBase64(null));
    }
}
