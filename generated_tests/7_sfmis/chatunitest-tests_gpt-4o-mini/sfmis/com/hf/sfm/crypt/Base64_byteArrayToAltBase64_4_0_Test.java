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
    public void testByteArrayToAltBase64() throws Exception {
        // Test cases for the byteArrayToAltBase64 method
        byte[] input1 = { 0, 1, 2, 3, 4, 5 };
        // Expected Base64 output for input1
        String expectedOutput1 = "AAECAwQFBg==";
        assertEquals(expectedOutput1, invokeByteArrayToAltBase64(input1));
        byte[] input2 = { -1, -2, -3, -4, -5 };
        // Expected Base64 output for input2
        String expectedOutput2 = "gID/";
        assertEquals(expectedOutput2, invokeByteArrayToAltBase64(input2));
        byte[] input3 = {};
        // Expected Base64 output for empty array
        String expectedOutput3 = "";
        assertEquals(expectedOutput3, invokeByteArrayToAltBase64(input3));
        byte[] input4 = { 10, 20, 30, 40, 50, 60, 70, 80, 90 };
        // Expected Base64 output for input4
        String expectedOutput4 = "ChQeHh8=";
        assertEquals(expectedOutput4, invokeByteArrayToAltBase64(input4));
    }

    private String invokeByteArrayToAltBase64(byte[] input) throws Exception {
        Method method = Base64.class.getDeclaredMethod("byteArrayToAltBase64", byte[].class);
        method.setAccessible(true);
        return (String) method.invoke(null, (Object) input);
    }
}
