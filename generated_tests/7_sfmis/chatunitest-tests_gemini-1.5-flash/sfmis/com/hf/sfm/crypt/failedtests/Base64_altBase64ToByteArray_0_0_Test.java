package com.hf.sfm.crypt;

import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_altBase64ToByteArray_0_0_Test {

    @Test
    void testAltBase64ToByteArray_validInput() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Base64 base64 = new Base64();
        Method method = Base64.class.getDeclaredMethod("_$23180", String.class, boolean.class);
        method.setAccessible(true);
        // "Hello World!" in Base64
        String validInput = "SGVsbG8gV29ybGQh";
        byte[] expectedOutput = "Hello World!".getBytes();
        byte[] actualOutput = (byte[]) method.invoke(base64, validInput, true);
        assertArrayEquals(expectedOutput, actualOutput);
        // "hello world!" in Base64
        String validInput2 = "aGVsbG8gd29ybGQh";
        byte[] expectedOutput2 = "hello world!".getBytes();
        byte[] actualOutput2 = (byte[]) method.invoke(base64, validInput2, true);
        assertArrayEquals(expectedOutput2, actualOutput2);
    }

    @Test
    void testAltBase64ToByteArray_invalidInput() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Base64 base64 = new Base64();
        Method method = Base64.class.getDeclaredMethod("_$23180", String.class, boolean.class);
        method.setAccessible(true);
        // "Hello World!" in Base64 with padding
        String invalidInput = "SGVsbG8gV29ybGQh==";
        byte[] actualOutput = (byte[]) method.invoke(base64, invalidInput, true);
        // The test will pass if it does not throw an exception.  The method handles padding.
        assertNotNull(actualOutput);
        // invalid character
        String invalidInput2 = "SGVsbG8gV29ybGQh%2B";
        assertThrows(IllegalArgumentException.class, () -> method.invoke(base64, invalidInput2, true));
        String invalidInput3 = null;
        assertThrows(NullPointerException.class, () -> method.invoke(base64, invalidInput3, true));
        String invalidInput4 = "";
        byte[] actualOutput4 = (byte[]) method.invoke(base64, invalidInput4, true);
        assertEquals(0, actualOutput4.length);
    }

    @Test
    void testAltBase64ToByteArray_emptyInput() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Base64 base64 = new Base64();
        Method method = Base64.class.getDeclaredMethod("_$23180", String.class, boolean.class);
        method.setAccessible(true);
        byte[] actualOutput = (byte[]) method.invoke(base64, "", true);
        assertEquals(0, actualOutput.length);
    }
}
