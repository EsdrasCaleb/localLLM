package com.hf.sfm.crypt;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class Base64_altBase64ToByteArray_0_2_Test {

    private Base64 base64;

    private Method privateMethod;

    @BeforeEach
    public void setUp() throws Exception {
        base64 = new Base64();
        privateMethod = Base64.class.getDeclaredMethod("_$23180", String.class, boolean.class);
        privateMethod.setAccessible(true);
    }

    @Test
    public void testAltBase64ToByteArray_EmptyString() throws Exception {
        String emptyBase64 = "";
        byte[] expectedOutput = new byte[0];
        byte[] actualOutput = base64.altBase64ToByteArray(emptyBase64);
        assertArrayEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testPrivateMethod_$23180_EmptyString() throws Exception {
        String emptyBase64 = "";
        byte[] expectedOutput = new byte[0];
        byte[] actualOutput = (byte[]) privateMethod.invoke(base64, emptyBase64, true);
        assertArrayEquals(expectedOutput, actualOutput);
    }
}
