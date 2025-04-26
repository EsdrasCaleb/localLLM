package com.hf.sfm.crypt;

import java.lang.reflect.Method;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_base64ToByteArray_2_0_Test {

    private Base64 base64;

    @BeforeEach
    public void setUp() {
        base64 = new Base64();
    }

    @Test
    public void testBase64ToByteArrayValidInput() throws Exception {
        // Test with a valid Base64 encoded string
        // "Hello World!" in Base64
        String validBase64 = "SGVsbG8gV29ybGQh";
        byte[] expectedOutput = "Hello World!".getBytes();
        byte[] result = Base64.base64ToByteArray(validBase64);
        assertTrue(Arrays.equals(expectedOutput, result));
    }

    @Test
    public void testBase64ToByteArrayInvalidInput() throws Exception {
        // Test with an invalid Base64 encoded string
        // "Hello World!" in Base64 with extra padding
        String invalidBase64 = "SGVsbG8gV29ybGQh==";
        Method method = Base64.class.getDeclaredMethod("_$23180", String.class, boolean.class);
        method.setAccessible(true);
        // Invoke the private method directly to check the behavior
        byte[] result = (byte[]) method.invoke(base64, invalidBase64, false);
        // Assuming the method returns null for invalid input
        assertNull(result);
    }

    @Test
    public void testBase64ToByteArrayEmptyString() throws Exception {
        // Test with an empty string
        String emptyBase64 = "";
        byte[] expectedOutput = new byte[0];
        byte[] result = Base64.base64ToByteArray(emptyBase64);
        assertTrue(Arrays.equals(expectedOutput, result));
    }

    @Test
    public void testBase64ToByteArrayNullInput() throws Exception {
        // Test with a null input
        String nullBase64 = null;
        Method method = Base64.class.getDeclaredMethod("_$23180", String.class, boolean.class);
        method.setAccessible(true);
        // Invoke the private method directly to check the behavior
        byte[] result = (byte[]) method.invoke(base64, nullBase64, false);
        // Assuming the method returns null for null input
        assertNull(result);
    }

    @Test
    public void testBase64ToByteArrayWithInvalidCharacters() throws Exception {
        // Test with a Base64 encoded string containing invalid characters
        // "Hello World!" in Base64 with an invalid character
        String invalidBase64 = "SGVsbG8gV29ybGQh*";
        Method method = Base64.class.getDeclaredMethod("_$23180", String.class, boolean.class);
        method.setAccessible(true);
        // Invoke the private method directly to check the behavior
        byte[] result = (byte[]) method.invoke(base64, invalidBase64, false);
        // Assuming the method returns null for invalid characters
        assertNull(result);
    }
}
