package com.hf.sfm.crypt;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_base64ToByteArray_2_0_Test {

    @Test
    public void testBase64ToByteArray_ValidInput() throws Exception {
        // "Hello World" in Base64
        String input = "SGVsbG8gV29ybGQ=";
        byte[] expected = "Hello World".getBytes();
        byte[] result = invokeBase64ToByteArray(input);
        assertArrayEquals(expected, result);
    }

    @Test
    public void testBase64ToByteArray_EmptyInput() throws Exception {
        String input = "";
        byte[] expected = new byte[0];
        byte[] result = invokeBase64ToByteArray(input);
        assertArrayEquals(expected, result);
    }

    @Test
    public void testBase64ToByteArray_InvalidInput() throws Exception {
        String input = "InvalidBase64@String!";
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            invokeBase64ToByteArray(input);
        });
        assertEquals("Invalid Base64 input", exception.getMessage());
    }

    private byte[] invokeBase64ToByteArray(String input) throws Exception {
        Method method = Base64.class.getDeclaredMethod("base64ToByteArray", String.class);
        method.setAccessible(true);
        return (byte[]) method.invoke(null, input);
    }
}
