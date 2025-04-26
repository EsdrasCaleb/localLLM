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
    void testAltBase64ToByteArray_emptyInput() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        Base64 base64 = new Base64();
        Method method = Base64.class.getDeclaredMethod("_$23180", String.class, boolean.class);
        method.setAccessible(true);
        byte[] actualOutput = (byte[]) method.invoke(base64, "", true);
        assertEquals(0, actualOutput.length);
    }
}
