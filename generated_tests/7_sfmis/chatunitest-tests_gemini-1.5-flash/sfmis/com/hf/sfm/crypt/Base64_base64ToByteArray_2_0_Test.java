package com.hf.sfm.crypt;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_base64ToByteArray_2_0_Test {

    @Test
    void testBase64ToByteArray_emptyString() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        String input = "";
        byte[] expected = new byte[0];
        byte[] actual = Base64.base64ToByteArray(input);
        assertArrayEquals(expected, actual);
    }

    @Test
    void testBase64ToByteArray_validString() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // "Hello World!" in Base64
        String input = "SGVsbG8gV29ybGQh";
        byte[] expected = "Hello World!".getBytes();
        byte[] actual = Base64.base64ToByteArray(input);
        assertArrayEquals(expected, actual);
    }

    @Test
    void testPrivateMethod_23180() throws Exception {
        Method method = Base64.class.getDeclaredMethod("_$23180", String.class, boolean.class);
        method.setAccessible(true);
        String input = "SGVsbG8gV29ybGQh";
        byte[] expected = "Hello World!".getBytes();
        byte[] actual = (byte[]) method.invoke(null, input, false);
        assertArrayEquals(expected, actual);
    }
}
