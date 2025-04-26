package com.densebrain.rif.util;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import org.apache.axis2.util.Base64;

public class ObjectUtility_encodeBytes_1_1_Test {

    @Test
    public void testEncodeBytes() throws Exception {
        byte[] input = "Hello, World!".getBytes();
        String expectedOutput = "SGVsbG8sIFdvcmxkIQ==";
        // Invoke the private encodeBytes method using reflection
        Method encodeBytesMethod = ObjectUtility.class.getDeclaredMethod("encodeBytes", byte[].class);
        encodeBytesMethod.setAccessible(true);
        String actualOutput = (String) encodeBytesMethod.invoke(null, (Object) input);
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testEncodeBytesWithNullInput() {
        assertThrows(NullPointerException.class, () -> {
            ObjectUtility.encodeBytes(null);
        });
    }

    @Test
    public void testEncodeBytesWithEmptyInput() {
        byte[] input = new byte[0];
        String expectedOutput = "";
        String actualOutput = ObjectUtility.encodeBytes(input);
        assertEquals(expectedOutput, actualOutput);
    }
}
