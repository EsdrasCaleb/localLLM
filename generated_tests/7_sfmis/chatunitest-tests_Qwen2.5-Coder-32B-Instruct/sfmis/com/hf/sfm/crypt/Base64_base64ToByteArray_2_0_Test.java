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
    public void testBase64ToByteArrayEmptyString() throws Exception {
        // Test with an empty string
        String emptyBase64 = "";
        byte[] expectedOutput = new byte[0];
        byte[] result = Base64.base64ToByteArray(emptyBase64);
        assertTrue(Arrays.equals(expectedOutput, result));
    }
}
