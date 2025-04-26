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

public class Base64_main_7_0_Test {

    @Test
    void testMainMethod() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Use reflection to access and invoke the private methods
        Class<?> base64Class = Base64.class;
        Method byteArrayToBase64Method = base64Class.getDeclaredMethod("byteArrayToBase64", byte[].class);
        Method base64ToByteArrayMethod = base64Class.getDeclaredMethod("base64ToByteArray", String.class);
        byteArrayToBase64Method.setAccessible(true);
        base64ToByteArrayMethod.setAccessible(true);
        String s = "0123456789";
        byte[] b = s.getBytes();
        // Test byteArrayToBase64
        String encoded = (String) byteArrayToBase64Method.invoke(null, b);
        assertNotNull(encoded);
        assertFalse(encoded.isEmpty());
        // Test base64ToByteArray
        byte[] decoded = (byte[]) base64ToByteArrayMethod.invoke(null, encoded);
        assertArrayEquals(b, decoded);
        String decodedString = new String(decoded);
        assertEquals(s, decodedString);
    }

    // Helper methods to access private fields if needed.  Not used in this specific test but included for completeness.
    private static Object getPrivateField(Object obj, String fieldName) throws NoSuchFieldException, IllegalAccessException {
        Field field = obj.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        return field.get(obj);
    }

    private static void setPrivateField(Object obj, String fieldName, Object value) throws NoSuchFieldException, IllegalAccessException {
        Field field = obj.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(obj, value);
    }
}
