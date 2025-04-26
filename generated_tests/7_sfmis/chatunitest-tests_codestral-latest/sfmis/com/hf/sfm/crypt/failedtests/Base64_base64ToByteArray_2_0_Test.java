package com.hf.sfm.crypt;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class Base64_base64ToByteArray_2_0_Test {

    @InjectMocks
    private Base64 base64;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testBase64ToByteArray() throws Exception {
        // Test with a valid Base64 encoded string
        // "Hello world!"
        String validBase64 = "SGVsbG8gd29ybGQh";
        byte[] expectedBytes = "Hello world!".getBytes();
        byte[] resultBytes = Base64.base64ToByteArray(validBase64);
        assertArrayEquals(expectedBytes, resultBytes);
        // Test with an empty string
        String emptyBase64 = "";
        byte[] emptyResultBytes = Base64.base64ToByteArray(emptyBase64);
        assertArrayEquals(new byte[0], emptyResultBytes);
        // Test with an invalid Base64 encoded string
        String invalidBase64 = "InvalidBase64String";
        assertThrows(IllegalArgumentException.class, () -> Base64.base64ToByteArray(invalidBase64));
        // Test with a null input
        assertThrows(NullPointerException.class, () -> Base64.base64ToByteArray(null));
        // Test with a string containing only whitespace
        String whitespaceBase64 = "   ";
        byte[] whitespaceResultBytes = Base64.base64ToByteArray(whitespaceBase64);
        assertArrayEquals(new byte[0], whitespaceResultBytes);
        // Test with a string containing padding characters
        // "Hello world!"
        String paddedBase64 = "SGVsbG8gd29ybGQh==";
        byte[] paddedResultBytes = Base64.base64ToByteArray(paddedBase64);
        assertArrayEquals(expectedBytes, paddedResultBytes);
    }

    @Test
    void testPrivateMethods() throws Exception {
        // Test the private method _$23180 using reflection
        Method privateMethod = Base64.class.getDeclaredMethod("_$23180", String.class, boolean.class);
        privateMethod.setAccessible(true);
        // Test with a valid Base64 encoded string
        // "Hello world!"
        String validBase64 = "SGVsbG8gd29ybGQh";
        byte[] expectedBytes = "Hello world!".getBytes();
        byte[] resultBytes = (byte[]) privateMethod.invoke(null, validBase64, false);
        assertArrayEquals(expectedBytes, resultBytes);
        // Test with an empty string
        String emptyBase64 = "";
        byte[] emptyResultBytes = (byte[]) privateMethod.invoke(null, emptyBase64, false);
        assertArrayEquals(new byte[0], emptyResultBytes);
        // Test with an invalid Base64 encoded string
        String invalidBase64 = "InvalidBase64String";
        assertThrows(IllegalArgumentException.class, () -> privateMethod.invoke(null, invalidBase64, false));
        // Test with a null input
        assertThrows(NullPointerException.class, () -> privateMethod.invoke(null, null, false));
        // Test with a string containing only whitespace
        String whitespaceBase64 = "   ";
        byte[] whitespaceResultBytes = (byte[]) privateMethod.invoke(null, whitespaceBase64, false);
        assertArrayEquals(new byte[0], whitespaceResultBytes);
        // Test with a string containing padding characters
        // "Hello world!"
        String paddedBase64 = "SGVsbG8gd29ybGQh==";
        byte[] paddedResultBytes = (byte[]) privateMethod.invoke(null, paddedBase64, false);
        assertArrayEquals(expectedBytes, paddedResultBytes);
    }

    @Test
    void testPrivateFields() throws Exception {
        // Test the private fields using reflection
        Field field23169 = Base64.class.getDeclaredField("_$23169");
        field23169.setAccessible(true);
        byte[] field23169Value = (byte[]) field23169.get(null);
        assertNotNull(field23169Value);
        Field field23168 = Base64.class.getDeclaredField("_$23168");
        field23168.setAccessible(true);
        byte[] field23168Value = (byte[]) field23168.get(null);
    }
}
